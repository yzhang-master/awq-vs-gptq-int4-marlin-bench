import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .marlin_adapter import (
    MarlinUnavailableError,
    build_marlin_model,
    marlin_storage_bytes,
    require_marlin,
)
from .marlin_data import (
    load_model_checkpoint,
    load_tokens,
    split_calibration_evaluation,
)
from .tiny import collect_linear_inputs, evaluate_perplexity, greedy_decode, linear_storage_bytes


@dataclass
class MarlinBenchConfig:
    model_checkpoint: Path
    dataset: Path
    output: Path
    method: str = "gptq"
    group_size: int = 128
    batch_size: int = 16
    decode_tokens: int = 128
    repeats: int = 50
    warmup: int = 2
    calibration_sequences: int = 64
    evaluation_batch_size: int = 32

    def validate(self) -> None:
        if self.method not in {"rtn", "awq", "gptq"}:
            raise ValueError("method must be rtn, awq, or gptq")
        if self.group_size not in {-1, 128}:
            raise ValueError("group_size must be -1 or 128 for official Marlin")
        for name in (
            "batch_size",
            "decode_tokens",
            "repeats",
            "calibration_sequences",
            "evaluation_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        inputs = {self.model_checkpoint.resolve(), self.dataset.resolve()}
        if self.output.resolve() in inputs:
            raise ValueError("output must not overwrite the checkpoint or token dataset")


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def measure_decode_cuda(
    model: nn.Module,
    initial: torch.Tensor,
    tokens: int,
    repeats: int,
    warmup: int,
    device: str,
) -> dict[str, float]:
    for _ in range(warmup):
        greedy_decode(model, initial, tokens)
    _synchronize(device)
    rates = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        greedy_decode(model, initial, tokens)
        _synchronize(device)
        elapsed = time.perf_counter() - start
        rates.append(initial.shape[0] * tokens / elapsed)
    return {
        "tokens_per_second": statistics.median(rates),
        "min_tokens_per_second": min(rates),
        "max_tokens_per_second": max(rates),
    }


def run_marlin_benchmark(
    config: MarlinBenchConfig,
    marlin_module=None,
    device: str = "cuda",
) -> dict:
    config.validate()
    model = load_model_checkpoint(config.model_checkpoint)
    tokens = load_tokens(config.dataset, model.vocab_size)
    calibration, evaluation = split_calibration_evaluation(tokens, config.calibration_sequences)
    if marlin_module is None:
        marlin_module = require_marlin()
    calibration_batches = [
        calibration[start : start + config.evaluation_batch_size, :-1]
        for start in range(0, calibration.shape[0], config.evaluation_batch_size)
    ]
    activations = collect_linear_inputs(model, calibration_batches)
    _synchronize(device)
    quantization_start = time.perf_counter()
    marlin_model = build_marlin_model(
        model,
        activations,
        config.method,
        config.group_size,
        marlin_module,
        device,
    )
    _synchronize(device)
    quantization_seconds = time.perf_counter() - quantization_start
    fp32_model = model.to(device)
    fp32_model.eval()
    marlin_model.eval()
    evaluation = evaluation.to(device)
    initial = evaluation[: config.batch_size, :1]
    if initial.shape[0] != config.batch_size:
        raise ValueError("evaluation split must contain at least batch_size sequences")

    fp32_perplexity = evaluate_perplexity(fp32_model, evaluation, config.evaluation_batch_size)
    marlin_perplexity = evaluate_perplexity(marlin_model, evaluation, config.evaluation_batch_size)
    fp32_rate = measure_decode_cuda(
        fp32_model,
        initial,
        config.decode_tokens,
        config.repeats,
        config.warmup,
        device,
    )
    marlin_rate = measure_decode_cuda(
        marlin_model,
        initial,
        config.decode_tokens,
        config.repeats,
        config.warmup,
        device,
    )
    result = {
        "status": "measured",
        "kernel": "official IST-DASLab marlin.Layer FP16xINT4 CUDA kernel",
        "config": {
            "model_checkpoint": str(config.model_checkpoint.resolve()),
            "dataset": str(config.dataset.resolve()),
            "method": config.method,
            "group_size": config.group_size,
            "batch_size": config.batch_size,
            "decode_tokens": config.decode_tokens,
            "repeats": config.repeats,
            "warmup": config.warmup,
            "calibration_sequences": config.calibration_sequences,
            "evaluation_sequences": evaluation.shape[0],
            "sequence_length": evaluation.shape[1],
            "device": device,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "marlin_version": getattr(marlin_module, "__version__", "0.1.1"),
            "marlin_module": getattr(marlin_module, "__file__", "unknown"),
            "gpu": torch.cuda.get_device_name(device)
            if device.startswith("cuda")
            else "cpu adapter test",
        },
        "quantization_seconds": quantization_seconds,
        "results": [
            {
                "method": "fp32",
                "perplexity": fp32_perplexity,
                **fp32_rate,
                "linear_storage_bytes": linear_storage_bytes(fp32_model),
            },
            {
                "method": f"marlin_{config.method}",
                "perplexity": marlin_perplexity,
                **marlin_rate,
                "linear_storage_bytes": marlin_storage_bytes(marlin_model),
                "speedup_vs_fp32": marlin_rate["tokens_per_second"]
                / fp32_rate["tokens_per_second"],
            },
        ],
        "storage_note": (
            "Marlin B and scale buffers, biases, and input scales; "
            "nonpersistent reduction workspaces, "
            "embeddings, and activations excluded"
        ),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark official Marlin INT4 against fp32")
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", choices=("rtn", "awq", "gptq"), default="gptq")
    parser.add_argument("--group-size", type=int, choices=(-1, 128), default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--calibration-sequences", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/marlin_cuda.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = MarlinBenchConfig(**vars(args))
    try:
        result = run_marlin_benchmark(config)
    except MarlinUnavailableError as error:
        parser.exit(2, f"{error}\n")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(2, f"ERROR: Marlin benchmark failed; no results written: {error}\n")
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
