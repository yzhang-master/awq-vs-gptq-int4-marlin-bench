import torch

from awq_gptq.quant import Int4Linear, PackedInt4, int4_matmul, quantize_int4


def test_nibble_layout_and_odd_width() -> None:
    weight = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    packed = quantize_int4(weight, group_size=5)
    assert packed.packed.dtype == torch.uint8
    assert packed.packed.shape == (1, 3)
    assert packed.unpack().shape == weight.shape
    reconstructed = packed.dequantize()
    assert reconstructed.shape == weight.shape
    assert torch.max(torch.abs(reconstructed - weight)) <= packed.scales.max() / 2 + 1e-6


def test_explicit_nibbles_decode_in_low_high_order() -> None:
    packed = PackedInt4(
        packed=torch.tensor([[0x21, 0x43]], dtype=torch.uint8),
        scales=torch.ones(1, 1),
        zero_points=torch.zeros(1, 1, dtype=torch.uint8),
        in_features=4,
        group_size=4,
    )
    assert packed.unpack().tolist() == [[1, 2, 3, 4]]


def test_constant_and_zero_groups_are_finite() -> None:
    weight = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, -3.0, -3.0]])
    packed = quantize_int4(weight, group_size=2)
    assert torch.allclose(packed.dequantize(), weight)
    assert torch.isfinite(packed.scales).all()


def test_group_scales_are_independent() -> None:
    weight = torch.tensor([[1e-4, -1e-4, 8.0, -8.0]])
    packed = quantize_int4(weight, group_size=2)
    assert packed.scales.shape == (1, 2)
    assert packed.scales[0, 1] / packed.scales[0, 0] > 10_000
    error = (packed.dequantize() - weight).abs().reshape(1, 2, 2).amax(dim=-1)
    assert (error <= packed.scales).all()


def test_int4_linear_uses_packed_storage_and_real_matmul() -> None:
    generator = torch.Generator().manual_seed(3)
    linear = torch.nn.Linear(7, 3)
    quantized = Int4Linear.from_linear(linear, group_size=4)
    x = torch.randn(2, 5, 7, generator=generator)
    expected = int4_matmul(x, quantized.weight) + linear.bias
    assert quantized.packed.shape == (3, 4)
    assert quantized.packed.dtype == torch.uint8
    assert torch.allclose(quantized(x), expected)
    assert not torch.allclose(quantized(x), linear(x))


def test_storage_counts_payload_scales_and_zero_points() -> None:
    packed = quantize_int4(torch.randn(8, 33), group_size=16)
    assert packed.storage_bytes == 8 * 17 + 8 * 3 * 4 + 8 * 3
