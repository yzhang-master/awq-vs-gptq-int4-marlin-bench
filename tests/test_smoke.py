import torch

from awq_gptq import __version__


def test_cpu_smoke() -> None:
    assert __version__ == "0.1.0"
    x = torch.tensor([[1.0, 2.0]])
    w = torch.tensor([[3.0], [4.0]])
    assert torch.equal(x @ w, torch.tensor([[11.0]]))
