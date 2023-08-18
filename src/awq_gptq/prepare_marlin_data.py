import argparse
from pathlib import Path

import torch

from .marlin_data import save_model_checkpoint
from .tiny import TinyBigramLM, make_markov_sequences, train_tiny_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a local Marlin-compatible checkpoint and tokens"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--calibration-sequences", type=int, default=64)
    parser.add_argument("--eval-sequences", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=33)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.train_sequences,
            args.calibration_sequences,
            args.eval_sequences,
            args.sequence_length,
        )
        <= 0
    ):
        parser.error("steps, sequence counts, and sequence length must be positive")
    torch.manual_seed(args.seed)
    train = make_markov_sequences(
        args.train_sequences,
        args.sequence_length,
        vocab_size=256,
        seed=101,
        transition_seed=303,
    )
    calibration = make_markov_sequences(
        args.calibration_sequences,
        args.sequence_length,
        vocab_size=256,
        seed=202,
        transition_seed=303,
    )
    evaluation = make_markov_sequences(
        args.eval_sequences,
        args.sequence_length,
        vocab_size=256,
        seed=404,
        transition_seed=303,
    )
    model = TinyBigramLM(vocab_size=256, hidden_size=256, expansion=2)
    losses = train_tiny_model(model, train, steps=args.steps)
    save_model_checkpoint(args.checkpoint, model)
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": torch.cat((calibration, evaluation))}, args.dataset)
    print(
        f"Wrote {args.checkpoint} and {args.dataset}; "
        f"training loss {losses[0]:.6f} -> {losses[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
