import json
from pathlib import Path

import pytest
import torch
from test_marlin_adapter import FakeMarlinModule

from awq_gptq.marlin_adapter import MarlinUnavailableError
from awq_gptq.marlin_bench import (
    MarlinBenchConfig,
    main,
    measure_decode_cuda,
    run_marlin_benchmark,
)
from awq_gptq.marlin_data import save_model_checkpoint
from awq_gptq.tiny import TinyBigramLM


def _config(tmp_path: Path) -> MarlinBenchConfig:
    return MarlinBenchConfig(
        model_checkpoint=tmp_path / "model.pt",
        dataset=tmp_path / "tokens.pt",
        output=tmp_path / "result.json",
        batch_size=2,
        decode_tokens=2,
        repeats=1,
        warmup=0,
        calibration_sequences=2,
        evaluation_batch_size=2,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("method", "fake"), ("group_size", 32), ("batch_size", 0), ("repeats", 0), ("warmup", -1)],
)
def test_benchmark_config_validation(tmp_path: Path, field: str, value) -> None:
    config = _config(tmp_path)
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_benchmark_cannot_overwrite_input(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output = config.dataset
    with pytest.raises(ValueError, match="overwrite"):
        config.validate()


def test_cpu_adapter_benchmark_exercises_model_eval_decode_and_json(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = TinyBigramLM(vocab_size=256, hidden_size=256)
    save_model_checkpoint(config.model_checkpoint, model)
    torch.save({"tokens": torch.randint(256, (6, 5))}, config.dataset)
    result = run_marlin_benchmark(config, FakeMarlinModule(), device="cpu")
    saved = json.loads(config.output.read_text())
    assert saved["status"] == "measured"
    assert len(saved["results"]) == 2
    assert all(row["perplexity"] > 0 for row in saved["results"])
    assert all(row["tokens_per_second"] > 0 for row in saved["results"])
    assert saved["results"][1]["method"] == "marlin_gptq"
    assert result["results"][1]["linear_storage_bytes"] > 0


def test_decode_timing_synchronizes_every_repetition(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    model = TinyBigramLM()
    initial = torch.tensor([[1], [2]])
    measure_decode_cuda(model, initial, tokens=2, repeats=3, warmup=1, device="cuda")
    assert len(calls) == 7


def test_cli_pending_failure_does_not_write_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def pending(_config):
        raise MarlinUnavailableError("PENDING: CUDA is unavailable; no output")

    monkeypatch.setattr("awq_gptq.marlin_bench.run_marlin_benchmark", pending)
    output = tmp_path / "pending.json"
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--model-checkpoint",
                str(tmp_path / "model.pt"),
                "--dataset",
                str(tmp_path / "tokens.pt"),
                "--output",
                str(output),
            ]
        )
    assert exit_info.value.code == 2
    assert "PENDING" in capsys.readouterr().err
    assert not output.exists()
