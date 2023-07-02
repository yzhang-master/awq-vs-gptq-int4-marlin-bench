import torch

from awq_gptq.awq import awq_linear, quantize_awq
from awq_gptq.quant import quantize_int4


def _calibration_case() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(6, 8, generator=generator) * 0.15
    weight[:, 0] = torch.linspace(-1.2, 1.2, 6)
    x = torch.randn(128, 8, generator=generator) * 0.05
    x[:, 0] = torch.randn(128, generator=generator) * 8
    return weight, x


def test_awq_scaling_reduces_salient_channel_output_error() -> None:
    weight, x = _calibration_case()
    baseline = quantize_int4(weight, group_size=4)
    result = quantize_awq(weight, x, group_size=4)
    reference = x @ weight.T
    baseline_mse = torch.mean((x @ baseline.dequantize().T - reference) ** 2)
    assert result.alpha > 0
    assert result.input_scale[0] > result.input_scale[1:].mean()
    assert result.calibration_mse < baseline_mse.item()


def test_awq_search_never_selects_worse_than_rtn_candidate() -> None:
    weight, x = _calibration_case()
    result = quantize_awq(weight, x, group_size=4, alphas=(0.0, 0.5))
    rtn = quantize_awq(weight, x, group_size=4, alphas=(0.0,))
    assert result.calibration_mse <= rtn.calibration_mse


def test_awq_linear_applies_inverse_activation_scale() -> None:
    weight, x = _calibration_case()
    linear = torch.nn.Linear(8, 6, bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    quantized = awq_linear(linear, x, group_size=4)
    scaled = quantized.input_scale
    expected = (x / scaled) @ quantized.weight.dequantize().T
    assert torch.allclose(quantized(x), expected)
