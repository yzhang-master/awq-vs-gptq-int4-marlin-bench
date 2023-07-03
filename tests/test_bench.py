import json
from pathlib import Path

from awq_gptq.bench import pareto_frontier, run_benchmark


def test_benchmark_emits_real_metrics_and_pareto_svg(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"
    result = run_benchmark(
        output,
        steps=2,
        train_sequences=16,
        eval_sequences=16,
        sequence_length=9,
        vocab_size=16,
        hidden_size=16,
        group_size=8,
        decode_tokens=2,
        decode_batch=2,
        repeats=1,
    )
    saved = json.loads(output.read_text())
    assert len(saved["results"]) == 4
    assert all(row["perplexity"] > 0 for row in saved["results"])
    assert all(row["tokens_per_second"] > 0 for row in saved["results"])
    assert output.with_name("smoke_pareto.svg").read_text().startswith("<svg")
    assert result["pareto_frontier"]


def test_pareto_frontier_discards_dominated_points() -> None:
    rows = [
        {"method": "a", "perplexity": 4.0, "tokens_per_second": 100.0},
        {"method": "b", "perplexity": 5.0, "tokens_per_second": 90.0},
        {"method": "c", "perplexity": 5.0, "tokens_per_second": 120.0},
    ]
    assert pareto_frontier(rows) == ["a", "c"]
