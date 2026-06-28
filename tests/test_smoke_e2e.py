from __future__ import annotations

import json
import tempfile
from pathlib import Path

from quantcall.backends.mock import MockBackend
from quantcall.config import load_config
from quantcall.datasets.smoke import load_smoke
from quantcall.manifest import write_manifest
from quantcall.runner import run_eval, write_result

SMOKE_CONFIG = Path(__file__).parent.parent / "configs" / "smoke.yaml"


def test_smoke_e2e_writes_result_json():
    cfg = load_config(SMOKE_CONFIG)
    instances = load_smoke()
    backend = MockBackend(latency_ms=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = Path(tmpdir) / "result.json"
        manifest_path = Path(tmpdir) / "manifest.json"

        result = run_eval(cfg, instances, backend, config_path=SMOKE_CONFIG)
        write_result(result, result_path)
        write_manifest(result.manifest, manifest_path)

        assert result_path.exists()
        assert manifest_path.exists()

        data = json.loads(result_path.read_text())
        manifest_data = json.loads(manifest_path.read_text())

        assert "svr" in data
        assert "tsa" in data
        assert "ac" in data
        assert "abstention" in data
        assert "fcr" in data
        assert "n" in data
        assert data["n"] == len(instances)
        assert "manifest" in data
        assert "config" in data

        assert "timestamp" in manifest_data
        assert "model" in manifest_data
        assert "backend" in manifest_data
        assert "config_sha256" in manifest_data
        assert "dataset_sha256" in manifest_data


def test_smoke_e2e_metrics_in_range():
    cfg = load_config(SMOKE_CONFIG)
    instances = load_smoke()
    backend = MockBackend(latency_ms=0)
    result = run_eval(cfg, instances, backend, config_path=SMOKE_CONFIG)

    assert 0.0 <= result.metrics.svr <= 1.0
    assert 0.0 <= result.metrics.tsa <= 1.0
    assert 0.0 <= result.metrics.ac <= 1.0
    assert 0.0 <= result.metrics.abstention <= 1.0
    assert 0.0 <= result.metrics.fcr <= 1.0


def test_smoke_e2e_all_instances_evaluated():
    cfg = load_config(SMOKE_CONFIG)
    instances = load_smoke()
    backend = MockBackend(latency_ms=0)
    result = run_eval(cfg, instances, backend, config_path=SMOKE_CONFIG)
    assert result.metrics.n == 10
    assert len(result.instance_results) == 10


def test_smoke_e2e_cli(smoke_config_path, tmp_path):
    from click.testing import CliRunner

    from quantcall.cli import main

    runner = CliRunner()
    out = tmp_path / "result.json"
    mf = tmp_path / "manifest.json"
    r = runner.invoke(
        main,
        [
            "run",
            "--config",
            str(smoke_config_path),
            "--output",
            str(out),
            "--manifest",
            str(mf),
        ],
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    assert mf.exists()
    data = json.loads(out.read_text())
    assert "svr" in data
