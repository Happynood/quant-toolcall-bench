from __future__ import annotations

from pathlib import Path

import pytest

from quantcall.config import QuantCallConfig, load_config

SMOKE_CONFIG = Path(__file__).parent.parent / "configs" / "smoke.yaml"


def test_smoke_config_loads():
    cfg = load_config(SMOKE_CONFIG)
    assert cfg.backend == "mock"
    assert cfg.model == "mock"
    assert "T0" in cfg.tiers
    assert cfg.seed == 42
    assert cfg.temperature == 0.0


def test_default_config_valid():
    cfg = QuantCallConfig()
    assert cfg.backend == "mock"
    assert cfg.sample_size >= 1
    assert cfg.repeats >= 1


def test_fcr_weights_sum_to_one():
    cfg = QuantCallConfig()
    w = cfg.metrics.fcr_weights
    total = w.svr + w.tsa + w.ac + w.abst
    assert abs(total - 1.0) < 1e-9


def test_config_model_copy():
    cfg = QuantCallConfig()
    cfg2 = cfg.model_copy(update={"backend": "llama-cpp", "quant": "Q4_K_M"})
    assert cfg2.backend == "llama-cpp"
    assert cfg2.quant == "Q4_K_M"
    assert cfg.backend == "mock"


def test_invalid_backend_raises():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QuantCallConfig.model_validate({"backend": "not-a-backend"})
