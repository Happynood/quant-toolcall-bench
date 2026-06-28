"""Tests for the quantcall suite build/materialize commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantcall.suite.build import (
    TIER_LICENSE_NOTES,
    TIER_SOURCE,
    build_suite,
)
from quantcall.suite.hashes import NORMALIZATION_VERSION, content_sha256
from quantcall.suite.manifest import ManifestEntry, SuiteManifest
from quantcall.suite.materialize import HashMismatchError, materialize_suite

# ── content_sha256 ────────────────────────────────────────────────────────────


def test_sha256_deterministic():
    """Same instance produces the same hash across two calls."""
    from quantcall.datasets.smoke import load_smoke

    inst = load_smoke()[0]
    h1 = content_sha256(inst)
    h2 = content_sha256(inst)
    assert h1 == h2
    assert len(h1) == 64  # hex SHA-256


def test_sha256_differs_on_change():
    """Changing query changes the hash."""
    from quantcall.datasets.base import NormalizedInstance
    from quantcall.datasets.smoke import load_smoke

    inst = load_smoke()[0]
    modified = NormalizedInstance(
        id=inst.id,
        tier=inst.tier,
        category=inst.category,
        query=inst.query + " MODIFIED",
        tools=inst.tools,
        ground_truth_calls=inst.ground_truth_calls,
        expects_call=inst.expects_call,
    )
    assert content_sha256(inst) != content_sha256(modified)


# ── ManifestEntry / SuiteManifest round-trip ─────────────────────────────────


def test_manifest_round_trip(tmp_path: Path):
    """SuiteManifest serialises and deserialises identically."""
    m = SuiteManifest(
        version="1",
        seed=42,
        sample_size_per_tier=10,
        normalization_version=NORMALIZATION_VERSION,
        created_at="2026-01-01T00:00:00+00:00",
        git_commit="abc123",
        tier_license_notes={"T0": "MIT"},
        entries=[
            ManifestEntry(
                id="T0-001",
                tier="T0",
                category="simple",
                source_repo="local:quantcall-smoke",
                source_revision="abc123",
                split="smoke",
                stable_index=0,
                seed=42,
                normalization_version=NORMALIZATION_VERSION,
                content_sha256="a" * 64,
            )
        ],
    )
    p = tmp_path / "manifest.json"
    m.save(p)
    loaded = SuiteManifest.load(p)
    assert loaded.version == m.version
    assert loaded.seed == m.seed
    assert len(loaded.entries) == 1
    assert loaded.entries[0].id == "T0-001"
    assert loaded.entries[0].content_sha256 == "a" * 64


# ── build_suite ────────────────────────────────────────────────────────────────


def test_build_t0_basic():
    """build_suite(['T0']) produces entries for all 10 smoke instances."""
    m = build_suite(tiers=["T0"], sample_size=100, seed=42)
    assert len(m.entries) == 10  # smoke set has exactly 10 instances
    for e in m.entries:
        assert e.tier == "T0"
        assert len(e.content_sha256) == 64
        assert e.normalization_version == NORMALIZATION_VERSION
        assert e.source_repo == "local:quantcall-smoke"


def test_build_t0_deterministic():
    """Two builds with the same seed produce identical manifests."""
    m1 = build_suite(tiers=["T0"], sample_size=100, seed=42)
    m2 = build_suite(tiers=["T0"], sample_size=100, seed=42)
    assert [e.content_sha256 for e in m1.entries] == [e.content_sha256 for e in m2.entries]
    assert [e.id for e in m1.entries] == [e.id for e in m2.entries]


def test_build_t0_seed_affects_sample():
    """Different seeds produce different samples when sample_size < total."""
    m1 = build_suite(tiers=["T0"], sample_size=5, seed=1)
    build_suite(tiers=["T0"], sample_size=5, seed=99)  # second call must not crash
    assert m1.sample_size_per_tier == 5
    assert len(m1.entries) == 5


def test_build_t0_license_notes_present():
    """License notes are recorded for each requested tier."""
    m = build_suite(tiers=["T0"], seed=42)
    assert "T0" in m.tier_license_notes
    assert "MIT" in m.tier_license_notes["T0"]


def test_build_gated_tiers_produce_no_entries():
    """T3/T4/T5 (non-redistributable) yield zero entries — no content embedded."""
    m = build_suite(tiers=["T3", "T4", "T5"], seed=42)
    assert len(m.entries) == 0
    # But license notes ARE recorded
    assert "T3" in m.tier_license_notes
    assert "T4" in m.tier_license_notes
    assert "T5" in m.tier_license_notes


def test_build_gated_license_notes_say_nc():
    """License notes for NC/gated tiers mention non-commercial or restricted."""
    m = build_suite(tiers=["T3", "T4", "T5"], seed=42)
    for tier in ("T3", "T4", "T5"):
        note = m.tier_license_notes[tier].lower()
        assert any(kw in note for kw in ("non-commercial", "gated", "nc", "unconfirmed")), (
            f"License note for {tier} does not mention restriction: {note!r}"
        )


def test_build_bfcl_without_data_dir_produces_no_entries():
    """T1 build without bfcl_data_dir silently skips (data not present)."""
    m = build_suite(tiers=["T1"], seed=42, bfcl_data_dir=None)
    assert len(m.entries_for_tier("T1")) == 0


def test_build_manifest_contains_no_raw_text_for_nc_tiers():
    """Manifest JSON for NC/gated tiers stores no query, tools, or answer content."""
    m = build_suite(tiers=["T3", "T4", "T5"], seed=42)
    raw = json.dumps(m.to_dict())
    # These should not appear in the manifest for NC tiers
    for kw in ("get_weather", "web_search", "search_flights"):
        assert kw not in raw, f"NC-tier manifest contains raw content keyword: {kw!r}"


# ── materialize_suite ────────────────────────────────────────────────────────


def test_materialize_t0_roundtrip():
    """Build T0 manifest then materialize — hashes match, instances identical."""
    m = build_suite(tiers=["T0"], sample_size=100, seed=42)
    instances = materialize_suite(m)
    assert len(instances) == 10
    # Verify each instance still hashes correctly
    for inst, entry in zip(instances, m.entries_for_tier("T0"), strict=True):
        assert content_sha256(inst) == entry.content_sha256


def test_materialize_determinism():
    """Materializing twice produces instances with identical hashes."""
    m = build_suite(tiers=["T0"], seed=42)
    hashes1 = [content_sha256(i) for i in materialize_suite(m)]
    hashes2 = [content_sha256(i) for i in materialize_suite(m)]
    assert hashes1 == hashes2


def test_materialize_hash_mismatch_raises():
    """Tampered hash in manifest raises HashMismatchError."""
    m = build_suite(tiers=["T0"], sample_size=1, seed=42)
    m.entries[0] = ManifestEntry(**{**m.entries[0].to_dict(), "content_sha256": "0" * 64})
    with pytest.raises(HashMismatchError):
        materialize_suite(m)


def test_materialize_nc_entries_raise():
    """Manifest with T4 entries (NC/gated) causes materialize to fail loudly."""
    m = build_suite(tiers=["T0"], sample_size=1, seed=42)
    # Manually inject a fake T4 entry to simulate a mistake
    fake_entry = ManifestEntry(
        id="T4-fake-001",
        tier="T4",
        category="xlam",
        source_repo="huggingface:Salesforce/xlam-function-calling-60k",
        source_revision="main",
        split="train",
        stable_index=0,
        seed=42,
        normalization_version=NORMALIZATION_VERSION,
        content_sha256="a" * 64,
    )
    m.entries.append(fake_entry)
    with pytest.raises(RuntimeError, match="T4"):
        materialize_suite(m)


# ── CLI integration ────────────────────────────────────────────────────────────


def test_suite_build_cli(tmp_path: Path):
    """CLI `suite build` writes a valid manifest file."""
    from click.testing import CliRunner

    from quantcall.cli import main

    out = tmp_path / "manifest.json"
    runner = CliRunner()
    result = runner.invoke(main, ["suite", "build", "--tiers", "T0", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    loaded = SuiteManifest.load(out)
    assert len(loaded.entries) == 10


def test_suite_materialize_cli(tmp_path: Path):
    """CLI `suite materialize` verifies T0 manifest and reports success."""
    from click.testing import CliRunner

    from quantcall.cli import main

    manifest_path = tmp_path / "manifest.json"
    runner = CliRunner()
    runner.invoke(main, ["suite", "build", "--tiers", "T0", "--output", str(manifest_path)])
    result = runner.invoke(main, ["suite", "materialize", "--manifest", str(manifest_path)])
    assert result.exit_code == 0, result.output
    assert "Verified 10 instances" in result.output


# ── HF suite artifact guard ───────────────────────────────────────────────────


def test_tier_source_constants_cover_all_tiers():
    """TIER_SOURCE and TIER_LICENSE_NOTES have entries for all defined tiers."""
    expected = {"T0", "T1", "T2", "T3", "T4", "T5", "T6"}
    assert expected <= set(TIER_SOURCE.keys())
    assert expected <= set(TIER_LICENSE_NOTES.keys())
