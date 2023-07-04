import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from .tiny import (
    TinyBigramLM,
    evaluate_perplexity,
    greedy_decode,
    linear_storage_bytes,
    make_markov_sequences,
    quantize_tiny_model,
    train_tiny_model,
)


def measure_decode(
    model: nn.Module, initial: torch.Tensor, tokens: int, repeats: int
) -> dict[str, float]:
    for _ in range(2):
        greedy_decode(model, initial, tokens)
    rates = []
    for _ in range(repeats):
        start = time.perf_counter()
        greedy_decode(model, initial, tokens)
        elapsed = time.perf_counter() - start
        rates.append(initial.shape[0] * tokens / elapsed)
    return {
        "tokens_per_second": statistics.median(rates),
        "min_tokens_per_second": min(rates),
        "max_tokens_per_second": max(rates),
    }


def pareto_frontier(rows: list[dict]) -> list[str]:
    frontier = []
    best_rate = float("-inf")
    for row in sorted(rows, key=lambda item: (item["perplexity"], -item["tokens_per_second"])):
        if row["tokens_per_second"] > best_rate:
            frontier.append(row["method"])
            best_rate = row["tokens_per_second"]
    return frontier


def write_pareto_svg(rows: list[dict], path: Path) -> None:
    width, height, margin = 720, 480, 70
    minimum_x = min(row["perplexity"] for row in rows)
    maximum_x = max(row["perplexity"] for row in rows)
    minimum_y = min(row["tokens_per_second"] for row in rows)
    maximum_y = max(row["tokens_per_second"] for row in rows)
    span_x = max(maximum_x - minimum_x, 1e-6)
    span_y = max(maximum_y - minimum_y, 1e-6)

    def point(row: dict) -> tuple[float, float]:
        x = margin + (row["perplexity"] - minimum_x) / span_x * (width - 2 * margin)
        y = (
            height
            - margin
            - (row["tokens_per_second"] - minimum_y) / span_y * (height - 2 * margin)
        )
        return x, y

    colors = {"fp32": "#222222", "rtn": "#d1495b", "awq": "#00798c", "gptq": "#6a4c93"}
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
            f'y2="{height - margin}" stroke="black"/>'
        ),
        (
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
            f'y2="{height - margin}" stroke="black"/>'
        ),
        (
            f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle">'
            "Perplexity (lower is better)</text>"
        ),
        (
            f'<text x="18" y="{height / 2}" text-anchor="middle" '
            f'transform="rotate(-90 18 {height / 2})">'
            "Greedy decode tokens/s (higher is better)</text>"
        ),
        (
            '<text x="360" y="30" text-anchor="middle" font-size="18">'
            "Tiny bigram LM on one CPU thread</text>"
        ),
    ]
    for row in rows:
        x, y = point(row)
        label = (
            f"{row['method']}: ppl={row['perplexity']:.3f}, {row['tokens_per_second']:.1f} tok/s"
        )
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{colors[row["method"]]}"/>')
        elements.append(f'<text x="{x + 9:.2f}" y="{y - 9:.2f}" font-size="12">{label}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n")


def run_benchmark(
    output: Path,
    steps: int = 160,
    train_sequences: int = 128,
    eval_sequences: int = 128,
    sequence_length: int = 33,
    vocab_size: int = 32,
    hidden_size: int = 32,
    group_size: int = 16,
    decode_tokens: int = 32,
    decode_batch: int = 16,
    repeats: int = 7,
    threads: int = 1,
) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(41)
    train = make_markov_sequences(
        train_sequences, sequence_length, vocab_size, seed=101, transition_seed=303
    )
    evaluation = make_markov_sequences(
        eval_sequences, sequence_length, vocab_size, seed=202, transition_seed=303
    )
    calibration = [
        train[start : start + 16, :-1] for start in range(0, min(64, train_sequences), 16)
    ]
    model = TinyBigramLM(vocab_size, hidden_size)
    training_losses = train_tiny_model(model, train, steps=steps)
    models = {"fp32": model}
    quantization_seconds = {}
    for method in ("rtn", "awq", "gptq"):
        start = time.perf_counter()
        models[method] = quantize_tiny_model(model, method, calibration, group_size)
        quantization_seconds[method] = time.perf_counter() - start
    initial = evaluation[:decode_batch, :1]
    rows = []
    for method, candidate in models.items():
        row = {
            "method": method,
            "perplexity": evaluate_perplexity(candidate, evaluation),
            **measure_decode(candidate, initial, decode_tokens, repeats),
            "linear_storage_bytes": linear_storage_bytes(candidate),
        }
        if method in quantization_seconds:
            row["quantization_seconds"] = quantization_seconds[method]
        rows.append(row)
    result = {
        "scope": (
            "Synthetic Markov bigram task; tiny trained torch model; "
            "explicit packed INT4 unpack + fp32 matmul on CPU"
        ),
        "config": {
            "seed": 41,
            "training_steps": steps,
            "train_sequences": train_sequences,
            "eval_sequences": eval_sequences,
            "sequence_length": sequence_length,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "group_size": group_size,
            "decode_tokens": decode_tokens,
            "decode_batch": decode_batch,
            "repeats": repeats,
            "threads": threads,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "training_loss_first": training_losses[0],
        "training_loss_last": training_losses[-1],
        "results": rows,
        "pareto_frontier": pareto_frontier(rows),
        "storage_note": (
            "Linear weights, scales, zero points, biases and explicit input-scale vectors; "
            "embeddings and activations excluded"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    write_pareto_svg(rows, output.with_name(f"{output.stem}_pareto.svg"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/tiny_cpu.json"))
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--eval-sequences", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=33)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--decode-batch", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    options = vars(args).copy()
    options.pop("smoke")
    if args.smoke:
        options.update(
            steps=2,
            train_sequences=16,
            eval_sequences=16,
            decode_tokens=2,
            decode_batch=2,
            repeats=1,
        )
    result = run_benchmark(**options)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
