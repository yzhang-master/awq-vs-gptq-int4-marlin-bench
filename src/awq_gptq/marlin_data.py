from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .marlin_adapter import validate_marlin_shape
from .tiny import TinyBigramLM

CHECKPOINT_FORMAT = "awq-gptq-tiny-bigram-v1"


def validate_marlin_config(config: dict) -> tuple[int, int, int]:
    if set(config) != {"vocab_size", "hidden_size", "expansion"}:
        raise ValueError("checkpoint config must contain vocab_size, hidden_size, and expansion")
    values = []
    for key in ("vocab_size", "hidden_size", "expansion"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"checkpoint {key} must be a positive integer")
        values.append(value)
    vocab_size, hidden_size, expansion = values
    validate_marlin_shape(hidden_size, hidden_size * expansion, 128)
    validate_marlin_shape(hidden_size * expansion, hidden_size, 128)
    validate_marlin_shape(hidden_size, vocab_size, 128)
    return vocab_size, hidden_size, expansion


def save_model_checkpoint(path: Path, model: TinyBigramLM) -> None:
    config = {
        "vocab_size": model.vocab_size,
        "hidden_size": model.embedding.embedding_dim,
        "expansion": model.fc1.out_features // model.embedding.embedding_dim,
    }
    validate_marlin_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format": CHECKPOINT_FORMAT, "config": config, "state_dict": model.state_dict()},
        path,
    )


def load_model_checkpoint(path: Path) -> TinyBigramLM:
    if path.suffix not in {".pt", ".pth"}:
        raise ValueError("model checkpoint must be a .pt or .pth file")
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format must be {CHECKPOINT_FORMAT}")
    if "config" not in payload or "state_dict" not in payload:
        raise ValueError("checkpoint must contain config and state_dict")
    vocab_size, hidden_size, expansion = validate_marlin_config(payload["config"])
    model = TinyBigramLM(vocab_size, hidden_size, expansion)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def load_tokens(path: Path, vocab_size: int) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"token dataset does not exist: {path}")
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("token .npy data must have an integer dtype")
        tokens = torch.from_numpy(array).to(torch.long)
    elif path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            payload = payload.get("tokens")
        if not isinstance(payload, Tensor) or payload.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("token .pt data must be an integer tensor or contain tokens")
        tokens = payload.to(torch.long)
    else:
        raise ValueError("token dataset must be .npy, .pt, or .pth")
    if tokens.ndim != 2 or tokens.shape[0] < 2 or tokens.shape[1] < 2:
        raise ValueError("tokens must have shape [sequences >= 2, sequence_length >= 2]")
    if tokens.min() < 0 or tokens.max() >= vocab_size:
        raise ValueError(f"token ids must be in [0, {vocab_size})")
    return tokens.contiguous()


def split_calibration_evaluation(
    tokens: Tensor, calibration_sequences: int
) -> tuple[Tensor, Tensor]:
    if not 0 < calibration_sequences < tokens.shape[0]:
        raise ValueError("calibration_sequences must leave at least one evaluation sequence")
    return tokens[:calibration_sequences], tokens[calibration_sequences:]
