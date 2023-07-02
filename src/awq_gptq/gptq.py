import torch
from torch import Tensor, nn

from .quant import Int4Linear, PackedInt4, _group_parameters


def _inverse_hessian(activations: Tensor, damping: float) -> Tensor:
    x = activations.detach().reshape(-1, activations.shape[-1]).to(torch.float64)
    if x.shape[0] == 0:
        raise ValueError("GPTQ requires at least one calibration activation")
    hessian = 2.0 * x.T @ x / x.shape[0]
    diagonal = torch.diagonal(hessian)
    hessian += (
        damping * diagonal.mean().clamp_min(1e-8) * torch.eye(hessian.shape[0], dtype=hessian.dtype)
    )
    chol = torch.linalg.cholesky(hessian)
    inverse = torch.cholesky_inverse(chol)
    return torch.linalg.cholesky(inverse, upper=True)


def quantize_gptq(
    weight: Tensor,
    activations: Tensor,
    group_size: int = 32,
    damping: float = 0.01,
) -> PackedInt4:
    if activations.shape[-1] != weight.shape[1]:
        raise ValueError("activations must match the weight input dimension")
    if damping <= 0:
        raise ValueError("damping must be positive")
    original = weight.detach().to(torch.float32).contiguous()
    updated = original.clone()
    out_features, in_features = original.shape
    groups = (in_features + group_size - 1) // group_size
    padded = torch.nn.functional.pad(original, (0, groups * group_size - in_features))
    blocks = padded.reshape(out_features, groups, group_size)
    scales, zeros = _group_parameters(blocks)
    inverse_hessian = _inverse_hessian(activations, damping).to(torch.float32)
    q = torch.empty_like(updated)

    for column in range(in_features):
        group = column // group_size
        scale = scales[:, group].unsqueeze(1)
        zero = zeros[:, group].unsqueeze(1)
        quantized = (updated[:, column : column + 1] / scale + zero).round().clamp(0, 15)
        q[:, column : column + 1] = quantized
        error = (updated[:, column : column + 1] - (quantized - zero) * scale) / inverse_hessian[
            column, column
        ].clamp_min(1e-8)
        if column + 1 < in_features:
            updated[:, column + 1 :] -= error * inverse_hessian[column, column + 1 :].unsqueeze(0)

    if in_features % 2:
        q = torch.nn.functional.pad(q, (0, 1))
    packed = q.to(torch.uint8)[:, 0::2] | (q.to(torch.uint8)[:, 1::2] << 4)
    return PackedInt4(packed, scales, zeros.to(torch.uint8), in_features, group_size)


def gptq_linear(
    linear: nn.Linear,
    activations: Tensor,
    group_size: int = 32,
    damping: float = 0.01,
) -> Int4Linear:
    packed = quantize_gptq(linear.weight, activations, group_size, damping)
    return Int4Linear(packed, linear.bias)
