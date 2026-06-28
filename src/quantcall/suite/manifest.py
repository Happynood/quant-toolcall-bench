from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifestEntry:
    """One sampled evaluation instance, described by provenance — no raw content."""

    id: str
    tier: str
    category: str
    source_repo: str
    source_revision: str
    split: str
    stable_index: int | str
    seed: int
    normalization_version: str
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tier": self.tier,
            "category": self.category,
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "split": self.split,
            "stable_index": self.stable_index,
            "seed": self.seed,
            "normalization_version": self.normalization_version,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManifestEntry:
        return cls(
            id=d["id"],
            tier=d["tier"],
            category=d["category"],
            source_repo=d["source_repo"],
            source_revision=d["source_revision"],
            split=d["split"],
            stable_index=d["stable_index"],
            seed=d["seed"],
            normalization_version=d["normalization_version"],
            content_sha256=d["content_sha256"],
        )


@dataclass
class SuiteManifest:
    version: str
    seed: int
    sample_size_per_tier: int
    normalization_version: str
    created_at: str = ""
    git_commit: str = ""
    entries: list[ManifestEntry] = field(default_factory=list)

    # Per-tier license decision (populated by build_suite).
    tier_license_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "sample_size_per_tier": self.sample_size_per_tier,
            "normalization_version": self.normalization_version,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "tier_license_notes": self.tier_license_notes,
            "entries": [e.to_dict() for e in self.entries],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> SuiteManifest:
        d = json.loads(Path(path).read_text())
        obj = cls(
            version=d["version"],
            seed=d["seed"],
            sample_size_per_tier=d["sample_size_per_tier"],
            normalization_version=d["normalization_version"],
            created_at=d.get("created_at", ""),
            git_commit=d.get("git_commit", ""),
            tier_license_notes=d.get("tier_license_notes", {}),
        )
        obj.entries = [ManifestEntry.from_dict(e) for e in d.get("entries", [])]
        return obj

    def entries_for_tier(self, tier: str) -> list[ManifestEntry]:
        return [e for e in self.entries if e.tier == tier]
