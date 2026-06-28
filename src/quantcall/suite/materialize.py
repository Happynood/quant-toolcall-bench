from __future__ import annotations

from pathlib import Path

from quantcall.datasets.base import NormalizedInstance
from quantcall.suite.hashes import content_sha256
from quantcall.suite.manifest import ManifestEntry, SuiteManifest


class HashMismatchError(ValueError):
    """Raised when a materialized instance does not match its manifest hash."""

    def __init__(self, entry_id: str, expected: str, actual: str) -> None:
        super().__init__(
            f"Hash mismatch for {entry_id!r}: expected {expected[:12]}… got {actual[:12]}…"
        )
        self.entry_id = entry_id
        self.expected = expected
        self.actual = actual


def _verify(inst: NormalizedInstance, entry: ManifestEntry) -> NormalizedInstance:
    actual = content_sha256(inst)
    if actual != entry.content_sha256:
        raise HashMismatchError(entry.id, entry.content_sha256, actual)
    return inst


def _materialize_t0(
    entries: list[ManifestEntry],
    smoke_path: str | Path | None = None,
) -> list[NormalizedInstance]:
    from quantcall.datasets.smoke import load_smoke

    all_instances = {inst.id: inst for inst in load_smoke(path=smoke_path)}
    result: list[NormalizedInstance] = []
    for entry in entries:
        inst = all_instances.get(entry.id)
        if inst is None:
            raise KeyError(f"T0 instance {entry.id!r} not found in smoke dataset")
        result.append(_verify(inst, entry))
    return result


def _materialize_bfcl(
    entries: list[ManifestEntry],
    bfcl_data_dir: str | Path,
    categories: list[str],
) -> list[NormalizedInstance]:
    from quantcall.datasets.bfcl import load_bfcl  # type: ignore[import]

    raw = load_bfcl(categories=categories, data_dir=str(bfcl_data_dir))
    all_instances = {inst.id: inst for inst in raw}
    result: list[NormalizedInstance] = []
    for entry in entries:
        inst = all_instances.get(entry.id)
        if inst is None:
            raise KeyError(f"BFCL instance {entry.id!r} not found")
        result.append(_verify(inst, entry))
    return result


def materialize_suite(
    manifest: SuiteManifest,
    smoke_path: str | Path | None = None,
    bfcl_data_dir: str | Path | None = None,
) -> list[NormalizedInstance]:
    """Reconstruct NormalizedInstances from a manifest; verify every SHA-256.

    Raises HashMismatchError on any hash mismatch.
    Raises ImportError for tiers that require optional deps not installed.
    Raises KeyError when a referenced instance cannot be located in the source.

    For T3/T4 (NC/gated tiers): no entries are expected in the manifest; raises
    RuntimeError if entries are somehow present to prevent silent redistribution.
    T5 (Apache 2.0) produces no entries yet because its adapter is not implemented,
    but entries would be allowed when the adapter is ready.
    """
    instances: list[NormalizedInstance] = []

    t0_entries = manifest.entries_for_tier("T0")
    if t0_entries:
        instances.extend(_materialize_t0(t0_entries, smoke_path))

    for tier, cats in [
        ("T1", ["simple", "multiple"]),
        ("T2", ["parallel", "parallel_multiple"]),
        ("T6", ["irrelevance"]),
    ]:
        tier_entries = manifest.entries_for_tier(tier)
        if tier_entries:
            if bfcl_data_dir is None:
                raise ValueError(f"Tier {tier} entries present but bfcl_data_dir not supplied")
            instances.extend(_materialize_bfcl(tier_entries, bfcl_data_dir, cats))

    # T3, T4: NC/gated — must not be redistributed. Fail loudly if entries present.
    # T5 is Apache 2.0; entries are allowed once its adapter is implemented.
    for tier in ("T3", "T4"):
        gated_entries = manifest.entries_for_tier(tier)
        if gated_entries:
            raise RuntimeError(
                f"Manifest contains {len(gated_entries)} entries for tier {tier} "
                f"({manifest.tier_license_notes.get(tier, 'unknown license')}). "
                "This tier is NC/gated and must not be redistributed. Remove these entries."
            )

    return instances
