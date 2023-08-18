import copy
import importlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .gptq import _inverse_hessian
from .quant import replace_linear


class MarlinUnavailableError(RuntimeError):
    pass


def load_marlin_module():
    try:
        module = importlib.import_module("marlin")
    except (ImportError, ModuleNotFoundError) as error:
        raise MarlinUnavailableError(
            "PENDING: official IST-DASLab Marlin is not installed; install its CUDA extension "
            "with `pip install .` from a Marlin source checkout. No benchmark output was written."
        ) from error
    if not hasattr(module, "Layer"):
        raise MarlinUnavailableError(
            "PENDING: the installed `marlin` package does not expose the official Layer API. "
            "No benchmark output was written."
        )
    return module


def require_marlin():
    if not torch.cuda.is_available():
        raise MarlinUnavailableError(
            "PENDING: CUDA is unavailable; the Marlin FP16xINT4 kernel requires an NVIDIA GPU "
            "with compute capability >= 8.0. No benchmark output was written."
        )
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise MarlinUnavailableError(
            f"PENDING: GPU compute capability {capability[0]}.{capability[1]} is unsupported; "
            "Marlin requires >= 8.0. No benchmark output was written."
        )
    return load_marlin_module()


def validate_marlin_shape(in_features: int, out_features: int, group_size: int) -> None:
    if in_features <= 0 or out_features <= 0:
        raise ValueError("Marlin dimensions must be positive")
    if in_features % 128 != 0 or out_features % 256 != 0:
        raise ValueError(
            "Official Marlin requires input features divisible by 128 and output features by 256"
        )
    if group_size not in {-1, 128}:
        raise ValueError("Official Marlin supports group_size -1 or 128")
    if group_size == 128 and in_features % 128 != 0:
        raise ValueError("input features must be divisible by group_size")


@dataclass
class SymmetricInt4:
    codes: Tensor
    dequantized: Tensor
    scales: Tensor
    input_scale: Tensor


def _symmetric_scales(weight: Tensor, group_size: int) -> Tensor:
    groups = (weight.shape[1] + group_size - 1) // group_size
    padded = F.pad(weight, (0, groups * group_size - weight.shape[1]))
    blocks = padded.reshape(weight.shape[0], groups, group_size)
    absmax = blocks.abs().amax(dim=-1)
    scales = 2.0 * absmax / 15.0
    return torch.where(scales > 0, scales, torch.ones_like(scales))


def quantize_symmetric_int4(weight: Tensor, group_size: int) -> SymmetricInt4:
    weight = weight.detach().to(torch.float32).contiguous()
    scales = _symmetric_scales(weight, group_size)
    expanded = scales.repeat_interleave(group_size, dim=1)[:, : weight.shape[1]]
    codes = torch.round(weight / expanded).clamp(-8, 7).to(torch.int8)
    return SymmetricInt4(
        codes=codes,
        dequantized=codes.to(torch.float32) * expanded,
        scales=scales,
        input_scale=torch.ones(weight.shape[1]),
    )


def quantize_symmetric_awq(
    weight: Tensor,
    activations: Tensor,
    group_size: int,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> SymmetricInt4:
    x = activations.detach().reshape(-1, activations.shape[-1]).to(torch.float32)
    if x.shape[0] == 0 or x.shape[1] != weight.shape[1]:
        raise ValueError("AWQ calibration activations do not match the linear layer")
    reference = x @ weight.detach().to(torch.float32).T
    salience = x.abs().mean(dim=0).clamp_min(1e-6)
    best: SymmetricInt4 | None = None
    best_mse = float("inf")
    for alpha in alphas:
        scale = salience.pow(alpha)
        scale = (scale / scale.mean().clamp_min(1e-12)).clamp_min(1e-4)
        candidate = quantize_symmetric_int4(weight * scale, group_size)
        reconstructed = (x / scale) @ candidate.dequantized.T
        mse = torch.mean((reconstructed - reference) ** 2).item()
        if mse < best_mse:
            best_mse = mse
            best = SymmetricInt4(candidate.codes, candidate.dequantized, candidate.scales, scale)
    if best is None:
        raise RuntimeError("AWQ search produced no candidate")
    return best


def quantize_symmetric_gptq(
    weight: Tensor,
    activations: Tensor,
    group_size: int,
    damping: float = 0.01,
) -> SymmetricInt4:
    original = weight.detach().to(torch.float32).contiguous()
    updated = original.clone()
    scales = _symmetric_scales(original, group_size)
    inverse_hessian = _inverse_hessian(activations, damping).to(torch.float32)
    codes = torch.empty_like(original)
    for column in range(original.shape[1]):
        scale = scales[:, column // group_size].unsqueeze(1)
        quantized = torch.round(updated[:, column : column + 1] / scale).clamp(-8, 7)
        codes[:, column : column + 1] = quantized
        error = (updated[:, column : column + 1] - quantized * scale) / inverse_hessian[
            column, column
        ].clamp_min(1e-8)
        if column + 1 < original.shape[1]:
            updated[:, column + 1 :] -= error * inverse_hessian[column, column + 1 :].unsqueeze(0)
    expanded = scales.repeat_interleave(group_size, dim=1)[:, : original.shape[1]]
    return SymmetricInt4(
        codes=codes.to(torch.int8),
        dequantized=codes * expanded,
        scales=scales,
        input_scale=torch.ones(original.shape[1]),
    )


def quantize_for_marlin(
    linear: nn.Linear,
    activations: Tensor,
    method: str,
    group_size: int,
) -> SymmetricInt4:
    actual_group_size = linear.in_features if group_size == -1 else group_size
    validate_marlin_shape(linear.in_features, linear.out_features, group_size)
    if method == "rtn":
        return quantize_symmetric_int4(linear.weight, actual_group_size)
    if method == "awq":
        return quantize_symmetric_awq(linear.weight, activations, actual_group_size)
    if method == "gptq":
        return quantize_symmetric_gptq(linear.weight, activations, actual_group_size)
    raise ValueError("method must be rtn, awq, or gptq")


class MarlinLinear(nn.Module):
    def __init__(
        self,
        kernel: nn.Module,
        bias: Tensor | None,
        input_scale: Tensor,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.register_buffer(
            "bias", None if bias is None else bias.detach().to(dtype=torch.float16)
        )
        self.register_buffer("input_scale", input_scale.detach().to(dtype=torch.float16))

    def forward(self, x: Tensor) -> Tensor:
        output = self.kernel(x / self.input_scale)
        if self.bias is not None:
            output = output + self.bias
        return output


def build_marlin_model(
    model: nn.Module,
    activations: dict[str, Tensor],
    method: str,
    group_size: int,
    marlin_module,
    device: str = "cuda",
) -> nn.Module:
    quantized = copy.deepcopy(model).to(device=device, dtype=torch.float16)
    for name, target in list(quantized.named_modules()):
        if not isinstance(target, nn.Linear):
            continue
        source = model.get_submodule(name)
        if name not in activations:
            raise ValueError(f"missing calibration activations for {name}")
        packed = quantize_for_marlin(source, activations[name], method, group_size)
        fake = nn.Linear(
            source.in_features,
            source.out_features,
            bias=False,
            device=device,
            dtype=torch.float16,
        )
        with torch.no_grad():
            fake.weight.copy_(packed.dequantized.to(device=device, dtype=torch.float16))
        kernel = marlin_module.Layer(
            source.in_features, source.out_features, groupsize=group_size
        ).to(device)
        kernel.pack(fake, packed.scales.to(device=device, dtype=torch.float16))
        replace_linear(
            quantized,
            name,
            MarlinLinear(kernel, target.bias, packed.input_scale.to(device)),
        )
    return quantized


def marlin_storage_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, MarlinLinear):
            total += module.kernel.B.nbytes + module.kernel.s.nbytes
            total += module.input_scale.nbytes
            if module.bias is not None:
                total += module.bias.nbytes
    return total
