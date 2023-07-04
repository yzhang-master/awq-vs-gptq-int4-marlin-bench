import copy
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .awq import awq_linear
from .gptq import gptq_linear
from .quant import Int4Linear, quantize_int4, replace_linear


class TinyBigramLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 32, expansion: int = 2) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * expansion)
        self.fc2 = nn.Linear(hidden_size * expansion, hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.embedding(tokens)
        hidden = hidden + self.fc2(F.gelu(self.fc1(self.norm1(hidden))))
        return self.head(self.norm2(hidden))


def make_markov_sequences(
    count: int,
    sequence_length: int,
    vocab_size: int = 32,
    branching: int = 3,
    seed: int = 123,
    transition_seed: int | None = None,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    transition_generator = torch.Generator().manual_seed(
        seed if transition_seed is None else transition_seed
    )
    successors = torch.randperm(vocab_size, generator=transition_generator)[:vocab_size]
    choices = torch.stack([successors.roll(int(index)) for index in range(branching)], dim=1)
    sequences = torch.empty(count, sequence_length, dtype=torch.long)
    sequences[:, 0] = torch.randint(vocab_size, (count,), generator=generator)
    for position in range(1, sequence_length):
        branch = torch.randint(branching, (count,), generator=generator)
        sequences[:, position] = choices[sequences[:, position - 1], branch]
    return sequences


def train_tiny_model(
    model: TinyBigramLM,
    sequences: Tensor,
    steps: int = 160,
    learning_rate: float = 0.01,
    seed: int = 77,
) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for _ in range(steps):
        indices = torch.randint(
            sequences.shape[0], (min(32, sequences.shape[0]),), generator=generator
        )
        batch = sequences[indices]
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate_perplexity(model: nn.Module, sequences: Tensor, batch_size: int = 32) -> float:
    model.eval()
    losses = []
    weights = []
    for start in range(0, sequences.shape[0], batch_size):
        batch = sequences[start : start + batch_size]
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten(), reduction="sum")
        losses.append(loss.item())
        weights.append(batch[:, 1:].numel())
    return float(torch.exp(torch.tensor(sum(losses) / sum(weights))))


@torch.no_grad()
def greedy_decode(model: nn.Module, initial: Tensor, tokens: int) -> Tensor:
    model.eval()
    generated = []
    current = initial
    for _ in range(tokens):
        current = model(current).argmax(dim=-1)
        generated.append(current)
    return torch.stack(generated, dim=1).squeeze(-1)


@torch.no_grad()
def collect_linear_inputs(model: nn.Module, batches: Iterable[Tensor]) -> dict[str, Tensor]:
    activations: dict[str, list[Tensor]] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):

            def capture(_module: nn.Module, inputs: tuple[Tensor, ...], key: str = name) -> None:
                activations.setdefault(key, []).append(
                    inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                )

            handles.append(module.register_forward_pre_hook(capture))
    try:
        for batch in batches:
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(parts, dim=0) for name, parts in activations.items()}


def quantize_tiny_model(
    model: TinyBigramLM,
    method: str,
    calibration: Iterable[Tensor],
    group_size: int = 16,
) -> TinyBigramLM:
    if method not in {"rtn", "awq", "gptq"}:
        raise ValueError("method must be rtn, awq, or gptq")
    quantized = copy.deepcopy(model)
    activations = collect_linear_inputs(quantized, calibration)
    for name, module in list(quantized.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if method == "awq":
            replacement = awq_linear(module, activations[name], group_size)
        elif method == "gptq":
            replacement = gptq_linear(module, activations[name], group_size)
        else:
            replacement = Int4Linear(quantize_int4(module.weight, group_size), module.bias)
        replace_linear(quantized, name, replacement)
    return quantized


def linear_storage_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.weight.nbytes
            if module.bias is not None:
                total += module.bias.nbytes
        elif isinstance(module, Int4Linear):
            total += module.packed.nbytes + module.scales.nbytes + module.zero_points.nbytes
            total += module.input_scale.nbytes
            if module.bias is not None:
                total += module.bias.nbytes
    return total
