from __future__ import annotations

import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quantcall.datasets.base import NormalizedInstance
from quantcall.datasets.smoke import load_smoke
from quantcall.suite.hashes import NORMALIZATION_VERSION, content_sha256
from quantcall.suite.manifest import ManifestEntry, SuiteManifest

MANIFEST_VERSION = "1"

# ---------------------------------------------------------------------------
# License decisions per tier
# ---------------------------------------------------------------------------
# Licenses were reviewed before populating this table.  Revise only when a
# source dataset's license changes; increment NORMALIZATION_VERSION when you do.
#
#  T0  – in-repo smoke set (MIT) — full redistribution permitted
#  T1/T2/T6 – BFCL (gorilla, Apache 2.0) — redistribution with attribution
#              permitted, but files require manual download so we record manifest
#              entries from local data when present; no HF upload of raw content
#  T3  – ToolACE (CC-BY-NC 4.0) — non-commercial; manifest-only, no redistrib
#  T4  – xLAM Salesforce (Research-only, non-commercial, gated) — manifest-only
#  T5  – Hermes function-calling (license unconfirmed) — manifest-only until
#         clarified; treat as non-redistributable
# ---------------------------------------------------------------------------

TIER_LICENSE_NOTES: dict[str, str] = {
    "T0": "MIT (in-repo). Full redistribution permitted.",
    "T1": "Apache 2.0 (gorilla/BFCL). Redistribution permitted with attribution. "
    "Files require manual download; manifest-only in HF artifact.",
    "T2": "Apache 2.0 (gorilla/BFCL). Same as T1.",
    "T3": "CC-BY-NC 4.0 (Team-ACE/ToolACE). Non-commercial — manifest-only, no redistribution.",
    "T4": "Research/non-commercial + gated (Salesforce/xlam-function-calling-60k). "
    "Manifest-only, no redistribution.",
    "T5": "License unconfirmed (teknium/hermes-function-calling-v1). "
    "Treated as non-redistributable until verified.",
    "T6": "Apache 2.0 (gorilla/BFCL). Same as T1.",
}

# Source identifiers used in manifest entries
TIER_SOURCE: dict[str, str] = {
    "T0": "local:quantcall-smoke",
    "T1": "gorilla-bfcl:simple+multiple",
    "T2": "gorilla-bfcl:parallel+parallel_multiple",
    "T3": "huggingface:Team-ACE/ToolACE",
    "T4": "huggingface:Salesforce/xlam-function-calling-60k",
    "T5": "huggingface:teknium/hermes-function-calling-v1",
    "T6": "gorilla-bfcl:irrelevance",
}


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return sha.decode().strip()
    except Exception:
        return "unknown"


def _sample(items: list[Any], n: int, seed: int) -> list[Any]:
    if len(items) <= n:
        return list(items)
    return random.Random(seed).sample(items, n)


def _entry_from_instance(
    inst: NormalizedInstance,
    *,
    source_repo: str,
    source_revision: str,
    split: str,
    stable_index: int | str,
    seed: int,
) -> ManifestEntry:
    return ManifestEntry(
        id=inst.id,
        tier=inst.tier,
        category=inst.category,
        source_repo=source_repo,
        source_revision=source_revision,
        split=split,
        stable_index=stable_index,
        seed=seed,
        normalization_version=NORMALIZATION_VERSION,
        content_sha256=content_sha256(inst),
    )


def _build_t0(sample_size: int, seed: int) -> list[ManifestEntry]:
    instances = load_smoke()
    sampled = _sample(instances, sample_size, seed)
    commit = _git_commit()
    return [
        _entry_from_instance(
            inst,
            source_repo="local:quantcall-smoke",
            source_revision=commit,
            split="smoke",
            stable_index=i,
            seed=seed,
        )
        for i, inst in enumerate(sampled)
    ]


def _build_bfcl(
    categories: list[str],
    tier: str,
    sample_size: int,
    seed: int,
    data_dir: Path | None,
) -> list[ManifestEntry]:
    if data_dir is None:
        return []
    try:
        from quantcall.datasets.bfcl import load_bfcl  # type: ignore[import]

        instances = load_bfcl(categories=categories, data_dir=str(data_dir))
    except (ImportError, FileNotFoundError):
        return []

    sampled = _sample(instances, sample_size, seed)
    source = TIER_SOURCE[tier]
    return [
        _entry_from_instance(
            inst,
            source_repo=source,
            source_revision="manual-download",
            split="+".join(categories),
            stable_index=i,
            seed=seed,
        )
        for i, inst in enumerate(sampled)
    ]


def build_suite(
    tiers: list[str] | None = None,
    sample_size: int = 100,
    seed: int = 42,
    bfcl_data_dir: str | Path | None = None,
) -> SuiteManifest:
    """Build a SuiteManifest by deterministically sampling each requested tier.

    For licensed/gated tiers (T3, T4, T5) this records provenance only — no
    content is embedded in the manifest.  For BFCL tiers (T1, T2, T6) content
    hashes are recorded if local data is present via bfcl_data_dir.
    """
    active_tiers = tiers or ["T0"]
    bfcl_dir = Path(bfcl_data_dir) if bfcl_data_dir else None

    manifest = SuiteManifest(
        version=MANIFEST_VERSION,
        seed=seed,
        sample_size_per_tier=sample_size,
        normalization_version=NORMALIZATION_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        git_commit=_git_commit(),
        tier_license_notes={t: TIER_LICENSE_NOTES.get(t, "Unknown") for t in active_tiers},
    )

    for tier in active_tiers:
        if tier == "T0":
            manifest.entries.extend(_build_t0(sample_size, seed))

        elif tier in ("T1", "T2", "T6"):
            cat_map = {
                "T1": ["simple", "multiple"],
                "T2": ["parallel", "parallel_multiple"],
                "T6": ["irrelevance"],
            }
            manifest.entries.extend(_build_bfcl(cat_map[tier], tier, sample_size, seed, bfcl_dir))

        elif tier in ("T3", "T4", "T5"):
            # Non-redistributable — record provenance placeholder only.
            # Actual hashes are filled in by materialize() when the user has
            # local access to the source dataset.
            manifest.tier_license_notes[tier] = TIER_LICENSE_NOTES.get(tier, "Unknown")
            # No entries added: callers must run materialize() with HF access.

    return manifest
