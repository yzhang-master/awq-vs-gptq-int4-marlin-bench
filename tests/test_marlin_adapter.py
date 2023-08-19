import importlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from awq_gptq.marlin_adapter import (
    MarlinLinear,
    MarlinUnavailableError,
    build_marlin_model,
    load_marlin_module,
    marlin_storage_bytes,
    quantize_symmetric_awq,
    quantize_symmetric_gptq,
    quantize_symmetric_int4,
    validate_marlin_shape,
)
from awq_gptq.tiny import TinyBigramLM, collect_linear_inputs


class FakeMarlinLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, groupsize: int = -1) -> None:
        super().__init__()
        validate_marlin_shape(in_features, out_features, groupsize)
        self.register_buffer(
            "B", torch.zeros(in_features // 16, out_features * 2, dtype=torch.int32)
        )
        actual_group = in_features if groupsize == -1 else groupsize
        self.register_buffer(
            "s", torch.zeros(in_features // actual_group, out_features, dtype=torch.float16)
        )

    def pack(self, linear: nn.Linear, scales: torch.Tensor) -> None:
        assert linear.weight.dtype == torch.float16
        self.reference = linear.weight.detach().clone()
        self.pack_scales = scales.detach().clone()
        self.B.fill_(1)
        self.s.copy_(scales.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.reference)


def test_symmetric_int4_has_real_codes_and_group_scales() -> None:
    weight = torch.randn(3, 10, generator=torch.Generator().manual_seed(8))
    packed = quantize_symmetric_int4(weight, group_size=5)
    assert packed.codes.dtype == torch.int8
    assert packed.codes.min() >= -8 and packed.codes.max() <= 7
    assert packed.scales.shape == (3, 2)
    assert packed.dequantized.shape == weight.shape
    error = (packed.dequantized - weight).abs().reshape(3, 2, 5).amax(dim=-1)
    assert (error <= packed.scales).all()


def test_symmetric_awq_uses_activation_salience() -> None:
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(6, 8, generator=generator) * 0.15
    weight[:, 0] = torch.linspace(-1.2, 1.2, 6)
    x = torch.randn(128, 8, generator=generator) * 0.05
    x[:, 0] = torch.randn(128, generator=generator) * 8
    packed = quantize_symmetric_awq(weight, x, group_size=4)
    assert packed.input_scale[0] > packed.input_scale[1:].mean()


def test_symmetric_gptq_compensates_correlated_inputs() -> None:
    generator = torch.Generator().manual_seed(11)
    latent = torch.randn(256, 2, generator=generator)
    mixing = torch.tensor([[1.0, 0.0], [0.95, 0.1], [0.9, -0.1], [0.0, 1.0]])
    x = latent @ mixing.T
    weight = torch.randn(5, 4, generator=generator)
    rtn = quantize_symmetric_int4(weight, 4)
    gptq = quantize_symmetric_gptq(weight, x, 4)
    reference = x @ weight.T
    assert not torch.equal(gptq.codes, rtn.codes)
    assert torch.mean((x @ gptq.dequantized.T - reference) ** 2) < torch.mean(
        (x @ rtn.dequantized.T - reference) ** 2
    )


def test_marlin_linear_applies_input_scale_and_bias() -> None:
    kernel = FakeMarlinLayer(128, 256, 128)
    linear = nn.Linear(128, 256, bias=False).half()
    kernel.pack(linear, torch.ones(256, 1))
    wrapped = MarlinLinear(kernel, torch.ones(256), torch.full((128,), 2.0))
    x = torch.randn(3, 128).half()
    assert torch.allclose(wrapped(x), F.linear(x / 2, linear.weight) + 1)


def test_build_model_packs_every_linear_through_marlin_adapter() -> None:
    torch.manual_seed(4)
    model = TinyBigramLM(vocab_size=256, hidden_size=256)
    tokens = torch.randint(256, (2, 5))
    activations = collect_linear_inputs(model, [tokens])
    quantized = build_marlin_model(model, activations, "rtn", 128, FakeMarlinModule(), device="cpu")
    wrapped = [module for module in quantized.modules() if isinstance(module, MarlinLinear)]
    assert len(wrapped) == 3
    assert all(module.kernel.B.dtype == torch.int32 for module in wrapped)
    assert all(module.kernel.pack_scales.shape[1] in {1, 2, 4} for module in wrapped)
    output = quantized(tokens)
    assert output.shape == (2, 5, 256)
    assert torch.isfinite(output).all()
    assert marlin_storage_bytes(quantized) > 0


class FakeMarlinModule:
    Layer = FakeMarlinLayer


def test_shape_validation_matches_official_marlin() -> None:
    validate_marlin_shape(128, 256, 128)
    validate_marlin_shape(128, 256, -1)
    with pytest.raises(ValueError, match="divisible"):
        validate_marlin_shape(64, 256, 128)
    with pytest.raises(ValueError, match="group_size"):
        validate_marlin_shape(128, 256, 32)


def test_missing_marlin_dependency_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("marlin")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(MarlinUnavailableError, match="PENDING"):
        load_marlin_module()
