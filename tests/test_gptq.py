import torch

from awq_gptq.gptq import gptq_linear, quantize_gptq
from awq_gptq.quant import quantize_int4


def _correlated_case() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    latent = torch.randn(256, 2, generator=generator)
    mixing = torch.tensor([[1.0, 0.0], [0.95, 0.1], [0.9, -0.1], [0.0, 1.0]])
    x = latent @ mixing.T
    weight = torch.randn(5, 4, generator=generator)
    return weight, x


def test_gptq_error_compensation_changes_naive_int4_codes() -> None:
    weight, x = _correlated_case()
    rtn = quantize_int4(weight, group_size=4)
    compensated = quantize_gptq(weight, x, group_size=4)
    assert compensated.packed.shape == rtn.packed.shape
    assert not torch.equal(compensated.packed, rtn.packed)


def test_gptq_improves_correlated_calibration_outputs() -> None:
    weight, x = _correlated_case()
    rtn = quantize_int4(weight, group_size=4)
    compensated = quantize_gptq(weight, x, group_size=4)
    reference = x @ weight.T
    rtn_mse = torch.mean((x @ rtn.dequantize().T - reference) ** 2)
    gptq_mse = torch.mean((x @ compensated.dequantize().T - reference) ** 2)
    assert gptq_mse < rtn_mse


def test_gptq_respects_multiple_group_boundaries() -> None:
    weight, x = _correlated_case()
    packed = quantize_gptq(weight, x, group_size=2)
    assert packed.scales.shape == (5, 2)
    assert packed.unpack().max() <= 15
    assert torch.isfinite(packed.dequantize()).all()


def test_gptq_linear_matches_packed_matrix_path() -> None:
    weight, x = _correlated_case()
    linear = torch.nn.Linear(4, 5)
    with torch.no_grad():
        linear.weight.copy_(weight)
    quantized = gptq_linear(linear, x, group_size=4)
    expected = x @ quantized.weight.dequantize().T + linear.bias
    assert torch.allclose(quantized(x), expected)
