from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .quant import Int4Linear, PackedInt4, quantize_int4


@dataclass
class AWQResult:
    weight: PackedInt4
    input_scale: Tensor
    alpha: float
    calibration_mse: float


def quantize_awq(
    weight: Tensor,
    activations: Tensor,
    group_size: int = 32,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    max_calibration_rows: int = 256,
) -> AWQResult:
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("alphas must be a non-empty sequence of non-negative values")
    x = activations.detach().reshape(-1, activations.shape[-1]).to(torch.float32)
    if x.shape[1] != weight.shape[1] or x.shape[0] == 0:
        raise ValueError("activations must be non-empty and match the weight input dimension")
    x = x[:max_calibration_rows]
    reference = x @ weight.detach().to(torch.float32).T
    salience = x.abs().mean(dim=0).clamp_min(1e-6)
    best: AWQResult | None = None
    for alpha in alphas:
        scale = salience.pow(alpha)
        scale = (scale / scale.mean().clamp_min(1e-12)).clamp_min(1e-4)
        packed = quantize_int4(weight * scale, group_size)
        reconstructed = (x / scale) @ packed.dequantize().T
        mse = torch.mean((reconstructed - reference) ** 2).item()
        if best is None or mse < best.calibration_mse:
            best = AWQResult(packed, scale, alpha, mse)
    if best is None:
        raise RuntimeError("AWQ search produced no candidate")
    return best


def awq_linear(
    linear: nn.Linear,
    activations: Tensor,
    group_size: int = 32,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Int4Linear:
    result = quantize_awq(linear.weight, activations, group_size, alphas)
    return Int4Linear(result.weight, linear.bias, result.input_scale)
