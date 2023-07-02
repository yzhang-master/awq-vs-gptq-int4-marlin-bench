from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PackedInt4:
    packed: Tensor
    scales: Tensor
    zero_points: Tensor
    in_features: int
    group_size: int

    @property
    def out_features(self) -> int:
        return self.packed.shape[0]

    @property
    def storage_bytes(self) -> int:
        return self.packed.nbytes + self.scales.nbytes + self.zero_points.nbytes

    def unpack(self) -> Tensor:
        low = self.packed & 0x0F
        high = self.packed >> 4
        return torch.stack((low, high), dim=-1).flatten(1)[:, : self.in_features]

    def dequantize(self) -> Tensor:
        q = self.unpack().to(self.scales.dtype)
        groups = torch.div(q.shape[1] + self.group_size - 1, self.group_size, rounding_mode="floor")
        q = F.pad(q, (0, groups * self.group_size - q.shape[1]))
        q = q.reshape(self.out_features, groups, self.group_size)
        return (
            (q - self.zero_points.to(q.dtype).unsqueeze(-1)) * self.scales.unsqueeze(-1)
        ).flatten(1)[:, : self.in_features]


def _group_parameters(block: Tensor) -> tuple[Tensor, Tensor]:
    minimum = block.amin(dim=-1)
    maximum = block.amax(dim=-1)
    scales = (maximum - minimum) / 15.0
    constant = scales < torch.finfo(scales.dtype).eps
    constant_scale = torch.where(
        minimum == 0,
        torch.ones_like(minimum),
        minimum.abs() / 15.0,
    )
    scales = torch.where(constant, constant_scale, scales)
    zero_points = torch.round(-minimum / scales).clamp(0, 15)
    negative_constant = constant & (minimum < 0)
    zero_points = torch.where(negative_constant, torch.full_like(zero_points, 15), zero_points)
    zero_points = torch.where(
        constant & ~negative_constant, torch.zeros_like(zero_points), zero_points
    )
    return scales, zero_points


def quantize_int4(weight: Tensor, group_size: int = 32) -> PackedInt4:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point [out_features, in_features] matrix")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    weight = weight.detach().to(torch.float32).contiguous()
    out_features, in_features = weight.shape
    groups = (in_features + group_size - 1) // group_size
    padded = F.pad(weight, (0, groups * group_size - in_features))
    blocks = padded.reshape(out_features, groups, group_size)
    scales, zero_points = _group_parameters(blocks)
    q = torch.round(blocks / scales.unsqueeze(-1)) + zero_points.unsqueeze(-1)
    q = q.clamp(0, 15).to(torch.uint8).reshape(out_features, -1)[:, :in_features]
    if in_features % 2:
        q = F.pad(q, (0, 1))
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return PackedInt4(
        packed=packed,
        scales=scales,
        zero_points=zero_points.to(torch.uint8),
        in_features=in_features,
        group_size=group_size,
    )


def int4_matmul(x: Tensor, weight: PackedInt4, input_scale: Tensor | None = None) -> Tensor:
    if x.shape[-1] != weight.in_features:
        raise ValueError("input feature dimension does not match packed weight")
    shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
    if input_scale is not None:
        flat = flat / input_scale.to(flat.dtype)
    output = flat @ weight.dequantize().T
    return output.reshape(*shape, weight.out_features)


class Int4Linear(nn.Module):
    def __init__(
        self,
        weight: PackedInt4,
        bias: Tensor | None = None,
        input_scale: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.register_buffer("packed", weight.packed)
        self.register_buffer("scales", weight.scales)
        self.register_buffer("zero_points", weight.zero_points)
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.float32))
        self.register_buffer(
            "input_scale",
            torch.ones(weight.in_features)
            if input_scale is None
            else input_scale.detach().to(torch.float32),
        )

    @property
    def in_features(self) -> int:
        return self.weight.in_features

    @property
    def out_features(self) -> int:
        return self.weight.out_features

    def packed_weight(self) -> PackedInt4:
        return PackedInt4(
            self.packed, self.scales, self.zero_points, self.in_features, self.weight.group_size
        )

    def forward(self, x: Tensor) -> Tensor:
        output = int4_matmul(x, self.packed_weight(), self.input_scale)
        if self.bias is not None:
            output = output + self.bias
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 32) -> "Int4Linear":
        return cls(quantize_int4(linear.weight, group_size), linear.bias)


def replace_linear(parent: nn.Module, name: str, replacement: nn.Module) -> None:
    components = name.split(".")
    module = parent
    for component in components[:-1]:
        module = module.get_submodule(component)
    setattr(module, components[-1], replacement)
