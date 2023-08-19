from pathlib import Path

import numpy as np
import pytest
import torch

from awq_gptq.marlin_data import (
    CHECKPOINT_FORMAT,
    load_model_checkpoint,
    load_tokens,
    save_model_checkpoint,
    split_calibration_evaluation,
    validate_marlin_config,
)
from awq_gptq.tiny import TinyBigramLM


def test_checkpoint_round_trip_is_strict_and_local(tmp_path: Path) -> None:
    model = TinyBigramLM(vocab_size=256, hidden_size=256)
    path = tmp_path / "model.pt"
    save_model_checkpoint(path, model)
    restored = load_model_checkpoint(path)
    assert restored.vocab_size == 256
    for original, loaded in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(original, loaded)


def test_checkpoint_rejects_incompatible_shapes(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": {"vocab_size": 32, "hidden_size": 32, "expansion": 2},
            "state_dict": TinyBigramLM().state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="divisible"):
        load_model_checkpoint(path)


def test_tokens_load_from_npy_and_pt(tmp_path: Path) -> None:
    tokens = np.arange(60, dtype=np.int32).reshape(6, 10) % 256
    npy = tmp_path / "tokens.npy"
    np.save(npy, tokens)
    assert torch.equal(load_tokens(npy, 256), torch.from_numpy(tokens).long())
    pt = tmp_path / "tokens.pt"
    torch.save({"tokens": torch.from_numpy(tokens)}, pt)
    assert torch.equal(load_tokens(pt, 256), torch.from_numpy(tokens).long())
    calibration, evaluation = split_calibration_evaluation(load_tokens(pt, 256), 2)
    assert calibration.shape == (2, 10)
    assert evaluation.shape == (4, 10)


@pytest.mark.parametrize("bad", [np.zeros((3, 4), dtype=np.float32), np.zeros(4, dtype=np.int64)])
def test_tokens_reject_float_or_wrong_rank(tmp_path: Path, bad: np.ndarray) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, bad)
    with pytest.raises(ValueError):
        load_tokens(path, 256)


def test_tokens_reject_out_of_vocabulary_id(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(torch.tensor([[0, 256], [1, 2]]), path)
    with pytest.raises(ValueError, match="token ids"):
        load_tokens(path, 256)


def test_marlin_checkpoint_config_is_explicit() -> None:
    assert validate_marlin_config({"vocab_size": 256, "hidden_size": 256, "expansion": 2}) == (
        256,
        256,
        2,
    )
    with pytest.raises(ValueError, match="config"):
        validate_marlin_config({"hidden_size": 256})
