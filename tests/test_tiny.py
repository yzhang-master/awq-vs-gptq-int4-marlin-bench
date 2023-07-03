import pytest
import torch

from awq_gptq.quant import Int4Linear
from awq_gptq.tiny import (
    TinyBigramLM,
    evaluate_perplexity,
    greedy_decode,
    linear_storage_bytes,
    make_markov_sequences,
    quantize_tiny_model,
    train_tiny_model,
)


def test_training_and_perplexity_use_next_token_targets() -> None:
    torch.manual_seed(5)
    sequences = make_markov_sequences(32, 12, vocab_size=16)
    model = TinyBigramLM(vocab_size=16, hidden_size=16)
    before = evaluate_perplexity(model, sequences)
    losses = train_tiny_model(model, sequences, steps=30, learning_rate=0.02)
    after = evaluate_perplexity(model, sequences)
    assert len(losses) == 30
    assert after < before


@pytest.mark.parametrize("method", ["rtn", "awq", "gptq"])
def test_quantized_tiny_model_evaluates_and_decodes(method: str) -> None:
    torch.manual_seed(9)
    sequences = make_markov_sequences(12, 9, vocab_size=16)
    model = TinyBigramLM(vocab_size=16, hidden_size=16)
    quantized = quantize_tiny_model(model, method, [sequences[:4, :-1]], group_size=8)
    assert all(not isinstance(module, torch.nn.Linear) for module in quantized.modules())
    assert sum(isinstance(module, Int4Linear) for module in quantized.modules()) == 3
    perplexity = evaluate_perplexity(quantized, sequences[:4])
    tokens = greedy_decode(quantized, sequences[:2, :1], tokens=5)
    assert torch.isfinite(torch.tensor(perplexity))
    assert tokens.shape == (2, 5)
    assert tokens.min() >= 0 and tokens.max() < 16
    assert linear_storage_bytes(quantized) < linear_storage_bytes(model)
