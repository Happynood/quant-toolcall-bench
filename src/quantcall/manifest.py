from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quantcall.config import QuantCallConfig
from quantcall.hardware import GpuInfo, collect_hardware


@dataclass(frozen=True)
class RunManifest:
    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    config_sha256: str
    dataset_sha256: str
    model: str
    backend: str
    quant: str
    decoding: str
    tiers: list[str]
    seed: int
    python_version: str
    platform_info: str
    cpu_model: str
    cpu_count: int | None
    gpu: GpuInfo | None


def collect_manifest(
    config_path: str | Path,
    cfg: QuantCallConfig,
    dataset_sha256: str = "",
) -> RunManifest:
    hw = collect_hardware()
    return RunManifest(
        timestamp=datetime.now(UTC).isoformat(),
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        config_sha256=_file_sha256(config_path),
        dataset_sha256=dataset_sha256,
        model=cfg.model,
        backend=cfg.backend,
        quant=cfg.quant,
        decoding=cfg.decoding,
        tiers=cfg.tiers,
        seed=cfg.seed,
        python_version=hw.python_version,
        platform_info=hw.platform_info,
        cpu_model=hw.cpu_model,
        cpu_count=hw.cpu_count,
        gpu=hw.gpu,
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def compute_dataset_sha256(instances: list[Any]) -> str:
    """Hash the IDs + content of all instances for reproducibility."""
    import json

    payload = json.dumps(
        [
            {
                "id": getattr(inst, "id", str(i)),
                "tier": getattr(inst, "tier", ""),
                "query": getattr(inst, "query", ""),
            }
            for i, inst in enumerate(instances)
        ],
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None
