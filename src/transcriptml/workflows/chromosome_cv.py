"""Immutable, example-balanced chromosome cross-validation plans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from transcriptml import __version__


PLAN_FORMAT = "transcriptml-chromosome-cv-plan"
PLAN_FORMAT_VERSION = "1"
PLAN_ALGORITHM = "largest_chromosome_first_greedy"


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", str(value))
        if part
    )


def _plan_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ChromosomeCVPlan:
    """A versioned assignment of complete chromosomes to fold groups."""

    n_folds: int
    group_col: str
    n_examples: int
    chromosome_counts: Mapping[str, int]
    fold_groups: tuple[tuple[str, ...], ...]
    fold_example_counts: tuple[int, ...]
    plan_id: str
    algorithm: str = PLAN_ALGORITHM
    transcriptml_version: str = __version__

    def _content_dict(self) -> dict[str, object]:
        return {
            "format": PLAN_FORMAT,
            "format_version": PLAN_FORMAT_VERSION,
            "n_folds": int(self.n_folds),
            "group_col": self.group_col,
            "n_examples": int(self.n_examples),
            "chromosome_counts": {
                chrom: int(self.chromosome_counts[chrom])
                for chrom in sorted(self.chromosome_counts, key=_natural_key)
            },
            "fold_groups": [
                {
                    "fold": fold,
                    "chromosomes": list(chromosomes),
                    "example_count": int(self.fold_example_counts[fold]),
                }
                for fold, chromosomes in enumerate(self.fold_groups)
            ],
            "generation": {
                "algorithm": self.algorithm,
                "group_order": "descending example count, then natural chromosome name",
                "fold_tie_break": "lowest current example count, then lowest fold index",
                "transcriptml_version": self.transcriptml_version,
            },
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete, self-validating plan."""

        payload = self._content_dict()
        payload["plan_id"] = self.plan_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ChromosomeCVPlan":
        """Validate and reconstruct a plan from JSON-like data."""

        if value.get("format") != PLAN_FORMAT:
            raise ValueError("not a TranscriptML chromosome CV plan")
        if str(value.get("format_version")) != PLAN_FORMAT_VERSION:
            raise ValueError("unsupported chromosome CV plan format version")
        raw_folds = value.get("fold_groups")
        if not isinstance(raw_folds, list):
            raise ValueError("chromosome CV plan fold_groups must be a list")
        folds: list[tuple[str, ...]] = []
        fold_counts: list[int] = []
        for expected_fold, raw in enumerate(raw_folds):
            if not isinstance(raw, Mapping) or int(raw.get("fold", -1)) != expected_fold:
                raise ValueError("chromosome CV plan folds must be consecutively indexed")
            chromosomes = raw.get("chromosomes")
            if not isinstance(chromosomes, list):
                raise ValueError("each chromosome CV fold must list chromosomes")
            folds.append(tuple(str(chrom) for chrom in chromosomes))
            fold_counts.append(int(raw.get("example_count", -1)))
        raw_counts = value.get("chromosome_counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError("chromosome CV plan lacks chromosome_counts")
        counts = {str(chrom): int(count) for chrom, count in raw_counts.items()}
        plan = cls(
            n_folds=int(value.get("n_folds", 0)),
            group_col=str(value.get("group_col", "")),
            n_examples=int(value.get("n_examples", -1)),
            chromosome_counts=counts,
            fold_groups=tuple(folds),
            fold_example_counts=tuple(fold_counts),
            plan_id=str(value.get("plan_id", "")),
            algorithm=str(
                value.get("generation", {}).get("algorithm", PLAN_ALGORITHM)
                if isinstance(value.get("generation"), Mapping)
                else PLAN_ALGORITHM
            ),
            transcriptml_version=str(
                value.get("generation", {}).get("transcriptml_version", "unknown")
                if isinstance(value.get("generation"), Mapping)
                else "unknown"
            ),
        )
        _validate_plan(plan)
        expected_id = _plan_digest(plan._content_dict())
        if plan.plan_id != expected_id:
            raise ValueError("chromosome CV plan_id does not match its contents")
        return plan


@dataclass(frozen=True)
class ChromosomeCVResolution:
    """Train/validation/test groups and row indices for one CV run."""

    fold: int
    validation_fold: int
    groups: Mapping[str, tuple[str, ...]]
    indices: Mapping[str, list[int]]


def _count_chromosomes(
    metadata: Sequence[Mapping[str, object]], group_col: str
) -> dict[str, int]:
    if not metadata:
        raise ValueError("metadata must contain at least one example")
    counts: dict[str, int] = {}
    for index, row in enumerate(metadata):
        value = row.get(group_col)
        if value is None or not str(value).strip():
            raise ValueError(
                f"metadata row {index} lacks chromosome grouping column {group_col!r}"
            )
        chromosome = str(value)
        counts[chromosome] = counts.get(chromosome, 0) + 1
    return counts


def _validate_plan(plan: ChromosomeCVPlan) -> None:
    if plan.n_folds < 3:
        raise ValueError("chromosome CV plans require at least three folds")
    if not plan.group_col:
        raise ValueError("chromosome CV group_col must be non-empty")
    if len(plan.fold_groups) != plan.n_folds:
        raise ValueError("chromosome CV fold count disagrees with n_folds")
    if len(plan.fold_example_counts) != plan.n_folds:
        raise ValueError("chromosome CV fold example counts disagree with n_folds")
    flattened = [chrom for group in plan.fold_groups for chrom in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("a chromosome occurs in more than one fold group")
    if set(flattened) != set(plan.chromosome_counts):
        raise ValueError("fold groups do not partition chromosome_counts")
    if any(int(count) <= 0 for count in plan.chromosome_counts.values()):
        raise ValueError("chromosome example counts must be positive")
    expected_fold_counts = tuple(
        sum(int(plan.chromosome_counts[chrom]) for chrom in chromosomes)
        for chromosomes in plan.fold_groups
    )
    if expected_fold_counts != plan.fold_example_counts:
        raise ValueError("fold example counts do not equal their chromosome totals")
    if sum(expected_fold_counts) != plan.n_examples:
        raise ValueError("chromosome CV plan example totals are inconsistent")
    if any(not chromosomes for chromosomes in plan.fold_groups):
        raise ValueError("every chromosome CV fold group must be non-empty")


def create_chromosome_cv_plan(
    metadata: Sequence[Mapping[str, object]],
    *,
    n_folds: int,
    group_col: str = "group_chromosome",
) -> ChromosomeCVPlan:
    """Greedily balance complete chromosomes by their example counts."""

    n_folds = int(n_folds)
    if n_folds < 3:
        raise ValueError("chromosome CV plans require at least three folds")
    counts = _count_chromosomes(metadata, str(group_col))
    if len(counts) < n_folds:
        raise ValueError(
            f"cannot create {n_folds} chromosome folds from only {len(counts)} chromosomes"
        )
    ordered = sorted(counts, key=lambda chrom: (-counts[chrom], _natural_key(chrom)))
    fold_groups: list[list[str]] = [[] for _ in range(n_folds)]
    fold_counts = [0] * n_folds
    for chromosome in ordered:
        fold = min(range(n_folds), key=lambda index: (fold_counts[index], index))
        fold_groups[fold].append(chromosome)
        fold_counts[fold] += counts[chromosome]
    normalized_groups = tuple(
        tuple(sorted(group, key=_natural_key)) for group in fold_groups
    )
    provisional = ChromosomeCVPlan(
        n_folds=n_folds,
        group_col=str(group_col),
        n_examples=len(metadata),
        chromosome_counts=dict(counts),
        fold_groups=normalized_groups,
        fold_example_counts=tuple(fold_counts),
        plan_id="",
    )
    _validate_plan(provisional)
    return ChromosomeCVPlan(
        **{
            **provisional.__dict__,
            "plan_id": _plan_digest(provisional._content_dict()),
        }
    )


def save_chromosome_cv_plan(
    plan: ChromosomeCVPlan, path: str | Path
) -> Path:
    """Write a stable human-readable chromosome CV plan JSON file."""

    _validate_plan(plan)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output


def load_chromosome_cv_plan(path: str | Path) -> ChromosomeCVPlan:
    """Load and validate an immutable chromosome CV plan."""

    return ChromosomeCVPlan.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def resolve_chromosome_cv_plan(
    plan: ChromosomeCVPlan,
    metadata: Sequence[Mapping[str, object]],
    *,
    fold: int,
) -> ChromosomeCVResolution:
    """Resolve one test fold, the following validation fold, and training rows."""

    fold = int(fold)
    if fold < 0 or fold >= plan.n_folds:
        raise ValueError(f"fold must be in [0, {plan.n_folds})")
    observed_counts = _count_chromosomes(metadata, plan.group_col)
    if observed_counts != dict(plan.chromosome_counts):
        raise ValueError(
            "dataset chromosome membership/counts differ from the saved CV plan"
        )
    validation_fold = (fold + 1) % plan.n_folds
    test_groups = plan.fold_groups[fold]
    validation_groups = plan.fold_groups[validation_fold]
    train_groups = tuple(
        chromosome
        for group_index, chromosomes in enumerate(plan.fold_groups)
        if group_index not in {fold, validation_fold}
        for chromosome in chromosomes
    )
    owner = {
        **{chrom: "train" for chrom in train_groups},
        **{chrom: "val" for chrom in validation_groups},
        **{chrom: "test" for chrom in test_groups},
    }
    indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for index, row in enumerate(metadata):
        indices[owner[str(row[plan.group_col])]].append(index)
    if sum(len(values) for values in indices.values()) != len(metadata):
        raise RuntimeError("chromosome CV resolution did not assign every example")
    return ChromosomeCVResolution(
        fold=fold,
        validation_fold=validation_fold,
        groups={
            "train": train_groups,
            "val": validation_groups,
            "test": test_groups,
        },
        indices=indices,
    )
