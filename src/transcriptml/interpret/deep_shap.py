"""Reference-aware DeepLIFT/SHAP attribution for RNA sequence models.

The workflow in this module deliberately separates reference construction,
reference evaluation, and attribution.  References are deterministic functions
of the original sequence index, which lets the evaluation and attribution
stages regenerate identical tensors without retaining a potentially enormous
``N x R x C x L`` array in memory.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from tangermeme import __version__ as tangermeme_version
from tangermeme.deep_lift_shap import deep_lift_shap, integrated_gradients_op
from tangermeme.ersatz import dinucleotide_shuffle, shuffle
from tangermeme.predict import predict as tangermeme_predict

from transcriptml.data.bundle import load_bundle
from transcriptml.data.region_edits import (
    base_channel_indices,
    base_symbols,
    region_bounds,
    resolve_cds_channel,
    valid_length_from_bases,
    write_base_symbols,
)
from transcriptml.data.schemas import SequenceSchema, get_schema
from transcriptml.interpret.backends import (
    TangermemeInput,
    _AnnotatedSequenceModel,
    _EnsembleModule,
    _predictor_module,
)
from transcriptml.interpret.codon_ism import (
    AA_TO_CODONS,
    CODON_TO_AA,
    SENSE_CODONS,
    STOP_CODONS,
    find_cds_codon_starts,
    resolve_analysis_indices,
)
from transcriptml.interpret.predictor import EnsemblePredictor, Predictor
from transcriptml.models.legnet import LegNet, SELayer
from transcriptml.models.reproduce import SalukiExact
from transcriptml.models.saluki import SalukiLike
from transcriptml.models.saluki_cnn import (
    SalukiCNNHybrid,
    _SalukiFeatureEncoder,
    _SmallCNNFeatureEncoder,
)
from transcriptml.models.universe import ConvolutionUniverse


ReferenceStrategy = Literal[
    "full-mono",
    "full-dinuc",
    "5utr-mono",
    "5utr-dinuc",
    "3utr-mono",
    "3utr-dinuc",
    "cds-codon-shuffle",
    "cds-synonymous",
    "cds-unrestricted",
]

REFERENCE_STRATEGIES: tuple[ReferenceStrategy, ...] = (
    "full-mono",
    "full-dinuc",
    "5utr-mono",
    "5utr-dinuc",
    "3utr-mono",
    "3utr-dinuc",
    "cds-codon-shuffle",
    "cds-synonymous",
    "cds-unrestricted",
)
_STRATEGY_CODES = {name: index + 1 for index, name in enumerate(REFERENCE_STRATEGIES)}


@dataclass(frozen=True)
class ReferenceConfig:
    """Configuration for one biologically defined reference distribution."""

    strategy: ReferenceStrategy = "full-mono"
    n_references: int = 20
    seed: int = 123
    preserve_initiation_codon: bool = True
    preserve_terminal_stop_codon: bool = True
    include_stop_codons: bool = False
    cds_channel: str | int | None = None

    def normalized(self) -> "ReferenceConfig":
        strategy = str(self.strategy).strip().lower().replace("_", "-")
        if strategy not in REFERENCE_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {', '.join(REFERENCE_STRATEGIES)}; got {self.strategy!r}"
            )
        if int(self.n_references) <= 0:
            raise ValueError("n_references must be positive")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        return ReferenceConfig(
            strategy=strategy,  # type: ignore[arg-type]
            n_references=int(self.n_references),
            seed=int(self.seed),
            preserve_initiation_codon=bool(self.preserve_initiation_codon),
            preserve_terminal_stop_codon=bool(self.preserve_terminal_stop_codon),
            include_stop_codons=bool(self.include_stop_codons),
            cds_channel=self.cds_channel,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self.normalized())


@dataclass(frozen=True)
class GeneratedReferences:
    """References and eligibility information for one input sequence."""

    references: np.ndarray | None
    eligible: bool
    reason: str | None
    valid_length: int
    mutable_start: int | None = None
    mutable_end: int | None = None


def _decode_codon(
    x: np.ndarray,
    start: int,
    *,
    base_channels: np.ndarray,
    base_letters: Sequence[str],
) -> str | None:
    symbols = base_symbols(x, int(start), int(start) + 3, base_channels)
    if symbols.shape != (3,) or np.any(symbols < 0):
        return None
    return "".join(base_letters[int(offset)] for offset in symbols)


def _write_codon(
    x: np.ndarray,
    start: int,
    codon: str,
    *,
    base_channels: np.ndarray,
    base_letters: Sequence[str],
) -> None:
    offsets = {base: index for index, base in enumerate(base_letters)}
    try:
        symbols = np.asarray([offsets[base] for base in codon], dtype=np.int16)
    except KeyError as exc:  # pragma: no cover - constants are validated by construction.
        raise ValueError(f"Invalid codon {codon!r}") from exc
    write_base_symbols(x, int(start), int(start) + 3, symbols, base_channels)


class ReferenceGenerator:
    """Generate deterministic full-shape references one sequence at a time."""

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        schema: str | SequenceSchema = "saluki6",
    ) -> None:
        self.config = (config or ReferenceConfig()).normalized()
        self.schema = get_schema(schema)
        self.base_channels = base_channel_indices(self.schema)
        self.base_letters = tuple(
            name.upper().replace("T", "U") for name in self.schema.base_channels
        )
        if tuple(self.base_channels.tolist()) != (0, 1, 2, 3):
            raise ValueError("DeepSHAP currently requires A/C/G/U to be the first four channels")
        if self.config.strategy not in {"full-mono", "full-dinuc"}:
            try:
                resolve_cds_channel(self.schema, self.config.cds_channel)
            except ValueError as exc:
                raise ValueError(
                    f"Reference strategy {self.config.strategy!r} requires a CDS annotation channel"
                ) from exc

    def _seed(self, sequence_index: int, reference_index: int) -> int:
        state = np.random.SeedSequence(
            [
                int(self.config.seed),
                int(sequence_index),
                int(_STRATEGY_CODES[self.config.strategy]),
                int(reference_index),
            ]
        ).generate_state(1, dtype=np.uint32)
        return int(state[0])

    def _cds(self, x: np.ndarray, valid_length: int):
        return find_cds_codon_starts(
            x,
            self.schema,
            valid_length=valid_length,
            cds_channel=self.config.cds_channel,
        )

    def _mutable_codon_starts(self, x: np.ndarray, cds) -> tuple[np.ndarray, set[int]]:
        starts = np.asarray(cds.starts, dtype=np.int64)
        starts = starts[(starts >= int(cds.cds_start)) & (starts + 2 <= int(cds.cds_end))]
        protected: set[int] = set()
        if starts.size and self.config.preserve_initiation_codon:
            protected.add(int(starts[0]))
        if starts.size and self.config.preserve_terminal_stop_codon:
            terminal = int(starts[-1])
            codon = _decode_codon(
                x,
                terminal,
                base_channels=self.base_channels,
                base_letters=self.base_letters,
            )
            if codon in STOP_CODONS:
                protected.add(terminal)
        return starts, protected

    def _shuffle_reference(
        self,
        x: np.ndarray,
        *,
        start: int,
        end: int,
        sequence_index: int,
        reference_index: int,
        dinucleotide: bool,
    ) -> np.ndarray:
        bases = torch.as_tensor(np.array(x[:4], copy=True)).unsqueeze(0)
        seed = self._seed(sequence_index, reference_index)
        if dinucleotide:
            shuffled = dinucleotide_shuffle(
                bases,
                start=int(start),
                end=int(end),
                n=1,
                random_state=seed,
                allow_N=True,
            )
        else:
            shuffled = shuffle(
                bases,
                start=int(start),
                end=int(end),
                n=1,
                random_state=seed,
            )
        out = np.array(x, copy=True)
        out[:4] = shuffled[0, 0].detach().cpu().numpy()
        return out

    def _codon_reference(
        self,
        x: np.ndarray,
        *,
        cds,
        sequence_index: int,
        reference_index: int,
    ) -> np.ndarray:
        out = np.array(x, copy=True)
        starts, protected = self._mutable_codon_starts(x, cds)
        editable = np.asarray([start for start in starts.tolist() if int(start) not in protected])
        rng = np.random.default_rng(self._seed(sequence_index, reference_index))
        strategy = self.config.strategy

        if strategy == "cds-codon-shuffle":
            codons = [
                np.array(base_symbols(x, int(start), int(start) + 3, self.base_channels), copy=True)
                for start in editable.tolist()
            ]
            if len(codons) > 1:
                codons = [codons[index] for index in rng.permutation(len(codons)).tolist()]
            for start, symbols in zip(editable.tolist(), codons, strict=True):
                write_base_symbols(out, int(start), int(start) + 3, symbols, self.base_channels)
            return out

        for start in editable.tolist():
            reference = _decode_codon(
                x,
                int(start),
                base_channels=self.base_channels,
                base_letters=self.base_letters,
            )
            if reference is None:
                continue
            if strategy == "cds-synonymous":
                amino_acid = CODON_TO_AA.get(reference)
                if amino_acid is None or (
                    amino_acid == "Stop" and not self.config.include_stop_codons
                ):
                    continue
                candidates = tuple(
                    codon for codon in AA_TO_CODONS[amino_acid] if codon != reference
                )
            else:
                pool = tuple(CODON_TO_AA) if self.config.include_stop_codons else SENSE_CODONS
                candidates = tuple(codon for codon in pool if codon != reference)
            if not candidates:
                continue
            alternate = candidates[int(rng.integers(0, len(candidates)))]
            _write_codon(
                out,
                int(start),
                alternate,
                base_channels=self.base_channels,
                base_letters=self.base_letters,
            )
        return out

    def generate(self, x: np.ndarray | torch.Tensor, *, sequence_index: int) -> GeneratedReferences:
        """Generate all configured references for one original input."""

        seq = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if seq.ndim != 2 or seq.shape[0] != self.schema.n_channels:
            raise ValueError(
                f"Expected one sequence with shape ({self.schema.n_channels}, L), got {seq.shape}"
            )
        valid_length = valid_length_from_bases(seq, self.base_channels)
        if valid_length <= 0:
            return GeneratedReferences(None, False, "empty_sequence", valid_length)

        strategy = self.config.strategy
        start, end = 0, valid_length
        cds = None
        if (
            strategy.startswith("5utr-")
            or strategy.startswith("3utr-")
            or strategy.startswith("cds-")
        ):
            cds = self._cds(seq, valid_length)
            if int(cds.cds_length) < 3 or np.asarray(cds.starts).size == 0:
                return GeneratedReferences(None, False, "missing_cds", valid_length)
        if strategy.startswith("5utr-"):
            bounds = region_bounds("5utr", valid_length=valid_length, cds=cds)
            assert bounds is not None
            start, end = bounds
        elif strategy.startswith("3utr-"):
            bounds = region_bounds("3utr", valid_length=valid_length, cds=cds)
            assert bounds is not None
            start, end = bounds
        elif strategy.startswith("cds-"):
            bounds = region_bounds("cds", valid_length=valid_length, cds=cds)
            assert bounds is not None
            start, end = bounds
        if end <= start:
            reason = "empty_5utr" if strategy.startswith("5utr-") else "empty_3utr"
            return GeneratedReferences(None, False, reason, valid_length, start, end)

        refs: list[np.ndarray] = []
        for reference_index in range(self.config.n_references):
            if strategy.endswith("-mono") or strategy.endswith("-dinuc"):
                refs.append(
                    self._shuffle_reference(
                        seq,
                        start=start,
                        end=end,
                        sequence_index=sequence_index,
                        reference_index=reference_index,
                        dinucleotide=strategy.endswith("-dinuc"),
                    )
                )
            else:
                assert cds is not None
                refs.append(
                    self._codon_reference(
                        seq,
                        cds=cds,
                        sequence_index=sequence_index,
                        reference_index=reference_index,
                    )
                )
        return GeneratedReferences(
            np.stack(refs, axis=0),
            True,
            None,
            valid_length,
            int(start),
            int(end),
        )


@dataclass
class ReferenceDiagnostic:
    """Sequence-level reference prediction and attribution diagnostics."""

    sequence_index: int
    sequence_id: str
    strategy: str
    eligible: bool
    eligibility_reason: str | None
    n_references: int
    original_prediction: float
    reference_prediction_mean: float | None = None
    reference_prediction_sd: float | None = None
    reference_prediction_min: float | None = None
    reference_prediction_max: float | None = None
    mean_reference_minus_original: float | None = None
    mean_absolute_prediction_difference: float | None = None
    fraction_above: float | None = None
    fraction_below: float | None = None
    fraction_near_zero: float | None = None
    maximum_directional_consistency: float | None = None
    effect_size_pass: bool | None = None
    confidence_classification: str = "ineligible"
    valid_length: int = 0
    attribution_sum: float | None = None
    expected_prediction_delta: float | None = None
    convergence_absolute_error: float | None = None
    convergence_relative_error: float | None = None
    convergence_pass: bool | None = None
    convergence_pairwise_warning: bool | None = None
    convergence_warning_count: int | None = None
    convergence_warnings: str | None = None
    ig_points_used: int | None = None


@dataclass
class ReferenceEvaluation:
    """Output of the reference-evaluation stage."""

    sequence_indices: np.ndarray
    diagnostics: list[ReferenceDiagnostic]
    original_predictions: np.ndarray
    high_confidence_indices: np.ndarray
    low_confidence_indices: np.ndarray
    ineligible_indices: np.ndarray
    confidence_threshold: float
    minimum_effect_size: float | None
    zero_atol: float

    @property
    def diagnostic_by_index(self) -> dict[int, ReferenceDiagnostic]:
        return {row.sequence_index: row for row in self.diagnostics}


def _predict_values(
    predictor: Predictor | EnsemblePredictor | Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    batch_size: int | None,
) -> np.ndarray:
    if hasattr(predictor, "predict"):
        try:
            values = predictor.predict(X, batch_size=batch_size)  # type: ignore[attr-defined]
        except TypeError:
            values = predictor.predict(X)  # type: ignore[attr-defined]
    else:
        values = predictor(X)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _predict_selected(
    predictor: Predictor | EnsemblePredictor | Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int | None,
) -> np.ndarray:
    """Predict selected rows without materializing a large fancy-indexed copy."""

    if len(indices) == 0:
        return np.empty(0, dtype=np.float32)
    chunk_size = int(batch_size or 128)
    if chunk_size <= 0:
        raise ValueError("batch_size must be positive")
    outputs: list[np.ndarray] = []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        values = _predict_values(predictor, X[chunk], batch_size=batch_size)
        if len(values) != len(chunk):
            raise ValueError(
                "Predictor returned a different number of predictions than input sequences"
            )
        outputs.append(values)
    return np.concatenate(outputs).astype(np.float32, copy=False)


class _ReferencePredictionWriter:
    _FIELDS = (
        "sequence_index",
        "sequence_id",
        "strategy",
        "reference_index",
        "prediction",
        "difference_from_original",
        "direction",
    )

    def __init__(self, path: str | Path | None, fmt: str) -> None:
        self.path = None if path is None else Path(path)
        self.format = str(fmt).lower()
        self.handle = None
        self.writer = None
        self.parquet_writer = None
        self.pa = None
        self.parquet_schema = None
        if self.format == "none":
            return
        if self.path is None:
            raise ValueError(
                "reference prediction output path is required when format is not 'none'"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.format == "csv":
            self.handle = self.path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.handle, fieldnames=self._FIELDS)
            self.writer.writeheader()
        elif self.format == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError(
                    "Parquet reference predictions require: pip install 'TranscriptML[arrow]'"
                ) from exc
            self.pa = pa
            schema = pa.schema(
                [
                    ("sequence_index", pa.int64()),
                    ("sequence_id", pa.string()),
                    ("strategy", pa.string()),
                    ("reference_index", pa.int64()),
                    ("prediction", pa.float32()),
                    ("difference_from_original", pa.float32()),
                    ("direction", pa.string()),
                ]
            )
            self.parquet_schema = schema
            self.parquet_writer = pq.ParquetWriter(self.path, schema)
        else:
            raise ValueError("reference_predictions_format must be one of: none, csv, parquet")

    def write(self, rows: list[dict[str, object]]) -> None:
        if not rows or self.format == "none":
            return
        if self.writer is not None:
            self.writer.writerows(rows)
        else:
            assert (
                self.pa is not None
                and self.parquet_writer is not None
                and self.parquet_schema is not None
            )
            self.parquet_writer.write_table(
                self.pa.Table.from_pylist(rows, schema=self.parquet_schema)
            )

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        if self.parquet_writer is not None:
            self.parquet_writer.close()


def evaluate_references(
    X: np.ndarray | torch.Tensor,
    predictor: Predictor | EnsemblePredictor | Callable[[np.ndarray], np.ndarray],
    reference_generator: ReferenceGenerator,
    *,
    sequence_ids: Sequence[str] | None = None,
    sequence_indices: Sequence[int] | None = None,
    batch_size: int | None = None,
    confidence_threshold: float = 0.95,
    minimum_effect_size: float | None = None,
    zero_atol: float = 1e-6,
    reference_predictions_path: str | Path | None = None,
    reference_predictions_format: Literal["none", "csv", "parquet"] = "none",
) -> ReferenceEvaluation:
    """Predict all references and classify each eligible original sequence."""

    arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    if arr.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {arr.shape}")
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if minimum_effect_size is not None and (
        not math.isfinite(float(minimum_effect_size)) or float(minimum_effect_size) < 0
    ):
        raise ValueError("minimum_effect_size must be finite and non-negative")
    if not math.isfinite(float(zero_atol)) or float(zero_atol) < 0:
        raise ValueError("zero_atol must be finite and non-negative")
    indices = (
        np.arange(arr.shape[0], dtype=np.int64)
        if sequence_indices is None
        else np.asarray(sequence_indices, dtype=np.int64)
    )
    if np.any(indices < 0) or np.any(indices >= arr.shape[0]):
        raise ValueError("sequence_indices contains indices outside X")
    ids = [str(i) for i in range(arr.shape[0])] if sequence_ids is None else list(sequence_ids)
    if len(ids) != arr.shape[0]:
        raise ValueError("sequence_ids length must match X.shape[0]")

    originals = _predict_selected(
        predictor,
        arr,
        indices,
        batch_size=batch_size,
    )
    writer = _ReferencePredictionWriter(reference_predictions_path, reference_predictions_format)
    diagnostics: list[ReferenceDiagnostic] = []
    high: list[int] = []
    low: list[int] = []
    ineligible: list[int] = []
    try:
        for local_index, sequence_index_value in enumerate(indices.tolist()):
            sequence_index = int(sequence_index_value)
            original = float(originals[local_index])
            generated = reference_generator.generate(
                arr[sequence_index], sequence_index=sequence_index
            )
            row = ReferenceDiagnostic(
                sequence_index=sequence_index,
                sequence_id=ids[sequence_index],
                strategy=reference_generator.config.strategy,
                eligible=generated.eligible,
                eligibility_reason=generated.reason,
                n_references=(
                    reference_generator.config.n_references if generated.eligible else 0
                ),
                original_prediction=original,
                valid_length=generated.valid_length,
            )
            if not generated.eligible:
                diagnostics.append(row)
                ineligible.append(sequence_index)
                continue

            assert generated.references is not None
            ref_predictions = _predict_values(
                predictor,
                generated.references,
                batch_size=batch_size,
            ).astype(np.float64)
            if len(ref_predictions) != reference_generator.config.n_references:
                raise ValueError(
                    "Predictor returned a different number of predictions than references"
                )
            differences = ref_predictions - original
            above = differences > float(zero_atol)
            below = differences < -float(zero_atol)
            near = ~(above | below)
            fraction_above = float(np.mean(above))
            fraction_below = float(np.mean(below))
            consistency = max(fraction_above, fraction_below)
            mean_absolute = float(np.mean(np.abs(differences)))
            effect_pass = (
                minimum_effect_size is None or mean_absolute >= float(minimum_effect_size)
            )
            classification = (
                "high_confidence"
                if consistency >= float(confidence_threshold) and effect_pass
                else "low_confidence"
            )
            row.reference_prediction_mean = float(np.mean(ref_predictions))
            row.reference_prediction_sd = float(np.std(ref_predictions, ddof=0))
            row.reference_prediction_min = float(np.min(ref_predictions))
            row.reference_prediction_max = float(np.max(ref_predictions))
            row.mean_reference_minus_original = float(np.mean(differences))
            row.mean_absolute_prediction_difference = mean_absolute
            row.fraction_above = fraction_above
            row.fraction_below = fraction_below
            row.fraction_near_zero = float(np.mean(near))
            row.maximum_directional_consistency = consistency
            row.effect_size_pass = effect_pass
            row.confidence_classification = classification
            diagnostics.append(row)
            (high if classification == "high_confidence" else low).append(sequence_index)

            prediction_rows = []
            for reference_index, (prediction, difference) in enumerate(
                zip(ref_predictions.tolist(), differences.tolist(), strict=True)
            ):
                direction = (
                    "above"
                    if difference > float(zero_atol)
                    else "below" if difference < -float(zero_atol) else "near_zero"
                )
                prediction_rows.append(
                    {
                        "sequence_index": sequence_index,
                        "sequence_id": ids[sequence_index],
                        "strategy": reference_generator.config.strategy,
                        "reference_index": reference_index,
                        "prediction": np.float32(prediction),
                        "difference_from_original": np.float32(difference),
                        "direction": direction,
                    }
                )
            writer.write(prediction_rows)
    finally:
        writer.close()

    return ReferenceEvaluation(
        sequence_indices=indices,
        diagnostics=diagnostics,
        original_predictions=originals.astype(np.float32, copy=False),
        high_confidence_indices=np.asarray(high, dtype=np.int64),
        low_confidence_indices=np.asarray(low, dtype=np.int64),
        ineligible_indices=np.asarray(ineligible, dtype=np.int64),
        confidence_threshold=float(confidence_threshold),
        minimum_effect_size=(None if minimum_effect_size is None else float(minimum_effect_size)),
        zero_atol=float(zero_atol),
    )


class _Output2D(torch.nn.Module):
    """Normalize a tensor model output to Tangermeme's two-dimensional form."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.model(X)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("DeepSHAP requires a tensor model output")
        if output.ndim == 1:
            return output.unsqueeze(1)
        if output.ndim == 2:
            return output
        trailing = int(np.prod(output.shape[2:]))
        if trailing == 1:
            return output.reshape(output.shape[0], output.shape[1])
        raise ValueError(
            "DeepSHAP requires one scalar per target and example; "
            f"model returned {tuple(output.shape)}"
        )


class _RecurrentSummary(torch.nn.Module):
    """Expose a recurrent layer plus its model-specific summary as one tensor op."""

    def __init__(
        self,
        recurrent: torch.nn.Module,
        *,
        summary: str,
        layers: int = 1,
        bidirectional: bool = False,
        recurrent_type: str = "gru",
    ) -> None:
        super().__init__()
        self.recurrent = recurrent
        self.summary = summary
        self.layers = int(layers)
        self.bidirectional = bool(bidirectional)
        self.recurrent_type = recurrent_type

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output, state = self.recurrent(X)
        if self.summary == "output-last":
            return output[:, -1, :]
        if self.summary == "mean":
            return output.mean(dim=1)
        if self.summary == "max":
            return output.amax(dim=1)
        hidden = state[0] if self.recurrent_type == "lstm" else state
        directions = 2 if self.bidirectional else 1
        hidden = hidden.view(self.layers, directions, hidden.shape[1], hidden.shape[2])
        final = hidden[-1]
        return final[0] if directions == 1 else torch.cat((final[0], final[1]), dim=1)


class _MaxMeanPool(torch.nn.Module):
    """Hookable global max-plus-mean pooling used by the hybrid CNN branch."""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat((X.amax(dim=-1), X.mean(dim=-1)), dim=1)


class _SalukiLikeAttributionModel(torch.nn.Module):
    def __init__(self, source: SalukiLike) -> None:
        super().__init__()
        self.encoder = source.encoder
        self.recurrent = _RecurrentSummary(
            source.gru,
            summary="state-last",
            layers=source.gru.num_layers,
            bidirectional=source.gru.bidirectional,
        )
        self.head = source.head

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(X.float()).transpose(1, 2)
        return self.head(self.recurrent(hidden)).squeeze(-1)


class _SalukiExactAttributionModel(torch.nn.Module):
    def __init__(self, source: SalukiExact) -> None:
        super().__init__()
        self.source = source
        self.recurrent = _RecurrentSummary(source.gru, summary="output-last")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        source = self.source
        z = source.shift(source._to_bcl(X))
        z = source.conv0(z)
        for block in source.blocks:
            z = block["ln"](z)
            z = block["act"](z)
            z = block["conv"](z)
            z = block["drop"](z)
            z = block["pool"](z)
        z = source.pre_rnn_act(source.pre_rnn_ln(z)).transpose(1, 2)
        z = self.recurrent(torch.flip(z, dims=[1]))
        z = source.fc1(torch.relu(source.bn1(z)))
        z = source.drop1(z)
        return source.fc2(torch.relu(source.bn2(z))).squeeze(-1)


class _SalukiFeatureAttributionModel(torch.nn.Module):
    def __init__(self, source: _SalukiFeatureEncoder) -> None:
        super().__init__()
        self.source = source
        self.recurrent = _RecurrentSummary(source.gru, summary="output-last")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        source = self.source
        z = source.conv0(X)
        for block in source.blocks:
            z = block["ln"](z)
            z = block["act"](z)
            z = block["conv"](z)
            z = block["drop"](z)
            z = block["pool"](z)
        z = source.pre_rnn_act(source.pre_rnn_ln(z)).transpose(1, 2)
        return self.recurrent(torch.flip(z, dims=[1]))


class _SmallCNNFeatureAttributionModel(torch.nn.Module):
    def __init__(self, source: _SmallCNNFeatureEncoder) -> None:
        super().__init__()
        self.encoder = source.encoder
        self.pool = _MaxMeanPool()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(X.float()))


class _SalukiCNNHybridAttributionModel(torch.nn.Module):
    def __init__(self, source: SalukiCNNHybrid) -> None:
        super().__init__()
        self.source = source
        self.saluki_encoder = _SalukiFeatureAttributionModel(source.saluki_encoder)
        self.cnn_encoder = _SmallCNNFeatureAttributionModel(source.cnn_encoder)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        source = self.source
        shared = source.shift(source._to_bcl(X))
        saluki = source.saluki_projection(self.saluki_encoder(shared))
        cnn = source.cnn_projection(self.cnn_encoder(shared))
        prediction = source.head(torch.cat((saluki, cnn), dim=1))
        return prediction[:, 0] if source.output_dim == 1 else prediction


class _ConvolutionUniverseAttributionModel(torch.nn.Module):
    def __init__(self, source: ConvolutionUniverse) -> None:
        super().__init__()
        if source.rnn is None:
            raise ValueError("The recurrent attribution adapter requires an RNN")
        self.source = source
        self.recurrent = _RecurrentSummary(
            source.rnn,
            summary=source.rnn_summary,
            layers=source.rnn_layers,
            bidirectional=source.rnn_bidirectional,
            recurrent_type=source.rnn_type,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        source = self.source
        z = source.shift(source._to_bcl(X))
        source.output_length(z.shape[-1])
        z = source.tower(z)
        z = source.pre_rnn_activation(source.pre_rnn_norm(z)).transpose(1, 2)
        if source.rnn_reverse:
            z = torch.flip(z, dims=[1])
        prediction = source.head(self.recurrent(z))
        return prediction[:, 0] if source.output_dim == 1 else prediction


def _adapt_attribution_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return a checkpoint-preserving model with hookable coupled operations."""

    if isinstance(model, SalukiLike):
        return _SalukiLikeAttributionModel(model)
    if isinstance(model, SalukiExact):
        return _SalukiExactAttributionModel(model)
    if isinstance(model, SalukiCNNHybrid):
        return _SalukiCNNHybridAttributionModel(model)
    if isinstance(model, ConvolutionUniverse) and model.rnn is not None:
        return _ConvolutionUniverseAttributionModel(model)
    return model


def _prepare_attribution_input(
    predictor: Predictor | EnsemblePredictor,
    X: np.ndarray | torch.Tensor,
    *,
    batch_size: int | None = None,
) -> TangermemeInput:
    module, device, default_batch_size, differentiable = _predictor_module(predictor)
    if not differentiable:
        raise TypeError("DeepSHAP requires differentiable PyTorch model predictors")
    if isinstance(module, _EnsembleModule):
        if module.reduction != "mean":
            raise ValueError("DeepSHAP supports mean ensembles but not median ensembles")
        module = _EnsembleModule(
            [_adapt_attribution_model(member) for member in module.models],
            "mean",
        )
    else:
        module = _adapt_attribution_model(module)
    module = _Output2D(module).eval()

    values = (
        X.detach().cpu()
        if isinstance(X, torch.Tensor)
        else torch.as_tensor(np.array(np.asarray(X), copy=True))
    )
    if values.ndim != 3 or values.shape[1] < 4:
        raise ValueError(f"Expected X with shape (N, C>=4, L), got {tuple(values.shape)}")
    args: tuple[torch.Tensor, ...] | None = None
    if values.shape[1] > 4:
        args = (values[:, 4:].contiguous(),)
        module = _AnnotatedSequenceModel(module)
    sequences = values[:, :4].contiguous()
    valid_values = torch.logical_or(sequences == 0, sequences == 1)
    column_sums = sequences.sum(dim=1)
    valid_columns = torch.logical_or(column_sums == 0, column_sums == 1)
    if not bool(torch.all(valid_values)) or not bool(torch.all(valid_columns)):
        raise ValueError(
            "DeepSHAP nucleotide channels must be one-hot encoded, with all-zero "
            "columns permitted for padding or unknown bases"
        )
    resolved_batch_size = int(default_batch_size if batch_size is None else batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return TangermemeInput(
        model=module,
        sequences=sequences,
        args=args,
        device=device,
        batch_size=resolved_batch_size,
        differentiable=True,
    )


class _TargetPredictor:
    """Prediction facade using the exact adapted model later attributed."""

    def __init__(
        self,
        predictor: Predictor | EnsemblePredictor,
        *,
        target: int,
        batch_size: int | None,
    ) -> None:
        module, device, default_batch_size, differentiable = _predictor_module(predictor)
        if not differentiable:
            raise TypeError("DeepSHAP requires differentiable PyTorch model predictors")
        if isinstance(module, _EnsembleModule):
            if module.reduction != "mean":
                raise ValueError("DeepSHAP supports mean ensembles but not median ensembles")
            module = _EnsembleModule(
                [_adapt_attribution_model(member) for member in module.models], "mean"
            )
        else:
            module = _adapt_attribution_model(module)
        self.model = _Output2D(module).eval()
        self.device = device
        self.batch_size = int(default_batch_size if batch_size is None else batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.target = int(target)

    def predict(self, X: np.ndarray | torch.Tensor, *, batch_size: int | None = None) -> np.ndarray:
        values = (
            X.detach().cpu()
            if isinstance(X, torch.Tensor)
            else torch.as_tensor(np.array(np.asarray(X), copy=True))
        )

        def select_target(output: torch.Tensor) -> torch.Tensor:
            if self.target < 0 or self.target >= output.shape[1]:
                raise ValueError(
                    f"target {self.target} is outside model output with {output.shape[1]} targets"
                )
            return output[:, self.target]

        resolved_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be positive")
        output = tangermeme_predict(
            self.model,
            values,
            func=select_target,
            batch_size=resolved_batch_size,
            dtype=torch.float32,
            device=self.device,
            verbose=False,
        )
        assert isinstance(output, torch.Tensor)
        return output.numpy().astype(np.float32, copy=False)


@dataclass
class DeepSHAPAttributionResult:
    """Partitioned nucleotide attribution tensors and their evaluation metadata."""

    high_confidence_hypothetical_attributions: np.ndarray
    high_confidence_actual_attributions: np.ndarray
    low_confidence_hypothetical_attributions: np.ndarray
    low_confidence_actual_attributions: np.ndarray
    evaluation: ReferenceEvaluation
    target: int
    ig_points: int
    max_ig_points: int
    convergence_atol: float
    convergence_rtol: float
    unresolved_convergence_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )


def _allocate_attributions(
    storage_dir: Path | None,
    name: str,
    shape: tuple[int, int, int],
) -> np.ndarray:
    if storage_dir is None or shape[0] == 0:
        return np.empty(shape, dtype=np.float32)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        storage_dir / f".{name}.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )


def _local_ig_operations(model: torch.nn.Module, points: int) -> dict[type, Callable]:
    operations: dict[type, Callable] = {}
    module_types = {type(module) for module in model.modules()}
    if _RecurrentSummary in module_types:
        operations[_RecurrentSummary] = integrated_gradients_op(K=points, name="recurrent")
    if SELayer in module_types:
        operations[SELayer] = integrated_gradients_op(K=points, name="squeeze_excitation")
    if _MaxMeanPool in module_types:
        operations[_MaxMeanPool] = integrated_gradients_op(K=points, name="max_mean_pool")
    return operations


def _is_convergence_warning(message: Warning) -> bool:
    text = str(message)
    return "Convergence deltas" in text or "of the output differences" in text


def _is_padding_warning(message: Warning) -> bool:
    text = str(message)
    return "X must be one-hot encoded" in text and "unknown characters" in text


def compute_deep_shap_attributions(
    X: np.ndarray | torch.Tensor,
    predictor: Predictor | EnsemblePredictor,
    reference_generator: ReferenceGenerator,
    evaluation: ReferenceEvaluation,
    *,
    target: int = 0,
    attribution_batch_size: int = 1,
    ig_points: int = 8,
    max_ig_points: int = 32,
    convergence_atol: float = 1e-3,
    convergence_rtol: float = 0.01,
    storage_dir: str | Path | None = None,
) -> DeepSHAPAttributionResult:
    """Calculate and partition DeepLIFT/SHAP attributions for eligible inputs."""

    arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    if arr.ndim != 3 or arr.shape[1] < 4:
        raise ValueError(f"Expected X with shape (N, C>=4, L), got {arr.shape}")
    if int(ig_points) <= 0 or int(max_ig_points) < int(ig_points):
        raise ValueError("ig_points must be positive and max_ig_points must be >= ig_points")
    if int(attribution_batch_size) <= 0:
        raise ValueError("attribution_batch_size must be positive")
    if float(convergence_atol) < 0 or float(convergence_rtol) < 0:
        raise ValueError("convergence tolerances must be non-negative")

    root = None if storage_dir is None else Path(storage_dir)
    shape_tail = (4, int(arr.shape[-1]))
    high_hyp = _allocate_attributions(
        root,
        "high_confidence_hypothetical_attributions",
        (len(evaluation.high_confidence_indices), *shape_tail),
    )
    high_actual = _allocate_attributions(
        root,
        "high_confidence_actual_attributions",
        (len(evaluation.high_confidence_indices), *shape_tail),
    )
    low_hyp = _allocate_attributions(
        root,
        "low_confidence_hypothetical_attributions",
        (len(evaluation.low_confidence_indices), *shape_tail),
    )
    low_actual = _allocate_attributions(
        root,
        "low_confidence_actual_attributions",
        (len(evaluation.low_confidence_indices), *shape_tail),
    )
    high_positions = {
        int(index): position for position, index in enumerate(evaluation.high_confidence_indices)
    }
    low_positions = {
        int(index): position for position, index in enumerate(evaluation.low_confidence_indices)
    }
    diagnostic_by_index = evaluation.diagnostic_by_index
    unresolved: list[int] = []

    for sequence_index_value in np.concatenate(
        (evaluation.high_confidence_indices, evaluation.low_confidence_indices)
    ):
        sequence_index = int(sequence_index_value)
        generated = reference_generator.generate(arr[sequence_index], sequence_index=sequence_index)
        if not generated.eligible or generated.references is None:
            raise RuntimeError(
                f"Sequence {sequence_index} changed eligibility between evaluation and attribution"
            )
        prepared = _prepare_attribution_input(
            predictor,
            arr[sequence_index : sequence_index + 1],
            batch_size=attribution_batch_size,
        )
        references = torch.as_tensor(np.asarray(generated.references[:, :4])).unsqueeze(0)
        diagnostic = diagnostic_by_index[sequence_index]
        expected_delta = float(
            diagnostic.original_prediction - float(diagnostic.reference_prediction_mean)
        )
        points = int(ig_points)
        final_hyp: np.ndarray | None = None
        final_actual: np.ndarray | None = None
        pairwise_warning = False
        pairwise_messages: list[str] = []
        all_pairwise_messages: list[str] = []
        convergence_pass = False
        absolute_error = math.inf
        relative_error = math.inf
        attribution_sum = math.nan
        while True:
            operations = _local_ig_operations(prepared.model, points)
            contains_padding = bool(torch.any(prepared.sequences.sum(dim=1) == 0))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                attributed = deep_lift_shap(
                    prepared.model,
                    prepared.sequences,
                    args=prepared.args,
                    target=int(target),
                    batch_size=int(attribution_batch_size),
                    references=references,
                    hypothetical=True,
                    warning_threshold=float(convergence_atol),
                    additional_nonlinear_ops=operations or None,
                    only_warn=contains_padding,
                    dtype=torch.float32,
                    device=prepared.device,
                    verbose=False,
                )
            pairwise_messages = [
                str(item.message) for item in caught if _is_convergence_warning(item.message)
            ]
            all_pairwise_messages.extend(
                f"K={points}: {message}" for message in pairwise_messages
            )
            pairwise_warning = bool(pairwise_messages)
            for item in caught:
                if not _is_convergence_warning(item.message) and not _is_padding_warning(
                    item.message
                ):
                    warnings.warn(str(item.message), item.category, stacklevel=2)
            hyp = attributed.detach().cpu().numpy()[0].astype(np.float32, copy=False)
            hyp = np.array(hyp, copy=True)
            hyp[:, generated.valid_length :] = 0
            actual = hyp * np.asarray(arr[sequence_index, :4], dtype=np.float32)
            attribution_sum = float(np.sum(actual, dtype=np.float64))
            absolute_error = abs(attribution_sum - expected_delta)
            tolerance = float(convergence_atol) + float(convergence_rtol) * abs(expected_delta)
            relative_error = absolute_error / max(
                abs(expected_delta), float(convergence_atol), 1e-12
            )
            convergence_pass = absolute_error <= tolerance and not pairwise_warning
            final_hyp, final_actual = hyp, actual
            if convergence_pass or points >= int(max_ig_points) or not operations:
                break
            points = min(points * 2, int(max_ig_points))

        assert final_hyp is not None and final_actual is not None
        diagnostic.attribution_sum = attribution_sum
        diagnostic.expected_prediction_delta = expected_delta
        diagnostic.convergence_absolute_error = absolute_error
        diagnostic.convergence_relative_error = relative_error
        diagnostic.convergence_pass = convergence_pass
        diagnostic.convergence_pairwise_warning = pairwise_warning
        diagnostic.convergence_warning_count = len(all_pairwise_messages)
        diagnostic.convergence_warnings = " | ".join(all_pairwise_messages) or None
        diagnostic.ig_points_used = points
        if not convergence_pass:
            unresolved.append(sequence_index)
        if sequence_index in high_positions:
            position = high_positions[sequence_index]
            high_hyp[position] = final_hyp
            high_actual[position] = final_actual
        else:
            position = low_positions[sequence_index]
            low_hyp[position] = final_hyp
            low_actual[position] = final_actual

    for array in (high_hyp, high_actual, low_hyp, low_actual):
        if isinstance(array, np.memmap):
            array.flush()
    if unresolved:
        warnings.warn(
            f"DeepSHAP convergence remained outside tolerance for {len(unresolved)} "
            f"sequence(s) after up to {max_ig_points} local-IG points; see diagnostics",
            RuntimeWarning,
            stacklevel=2,
        )
    return DeepSHAPAttributionResult(
        high_confidence_hypothetical_attributions=high_hyp,
        high_confidence_actual_attributions=high_actual,
        low_confidence_hypothetical_attributions=low_hyp,
        low_confidence_actual_attributions=low_actual,
        evaluation=evaluation,
        target=int(target),
        ig_points=int(ig_points),
        max_ig_points=int(max_ig_points),
        convergence_atol=float(convergence_atol),
        convergence_rtol=float(convergence_rtol),
        unresolved_convergence_indices=np.asarray(unresolved, dtype=np.int64),
    )


def _save_array(path: Path, array: np.ndarray) -> np.ndarray:
    """Atomically publish an array, moving completed memory maps when possible."""

    filename = getattr(array, "filename", None)
    if filename is not None:
        source = Path(filename)
        if isinstance(array, np.memmap):
            array.flush()
        if source.resolve() == path.resolve():
            return array
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
        os.replace(source, path)
        return np.load(path, mmap_mode="r+")

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp.npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, np.asarray(array))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return array


def _atomic_write_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_deep_shap_result(
    result: DeepSHAPAttributionResult,
    out_dir: str | Path,
    *,
    reference_config: ReferenceConfig,
    sequence_ids: Sequence[str],
    checkpoint: str | Path | None = None,
    dataset: str | Path | None = None,
    reference_predictions_format: str = "none",
) -> None:
    """Persist partitioned attribution tensors and their complete provenance."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluation = result.evaluation
    result.high_confidence_hypothetical_attributions = _save_array(
        out / "high_confidence_hypothetical_attributions.npy",
        result.high_confidence_hypothetical_attributions,
    )
    result.high_confidence_actual_attributions = _save_array(
        out / "high_confidence_actual_attributions.npy",
        result.high_confidence_actual_attributions,
    )
    result.low_confidence_hypothetical_attributions = _save_array(
        out / "low_confidence_hypothetical_attributions.npy",
        result.low_confidence_hypothetical_attributions,
    )
    result.low_confidence_actual_attributions = _save_array(
        out / "low_confidence_actual_attributions.npy",
        result.low_confidence_actual_attributions,
    )
    arrays = {
        "high_confidence_hypothetical_attributions": (
            result.high_confidence_hypothetical_attributions
        ),
        "high_confidence_actual_attributions": result.high_confidence_actual_attributions,
        "low_confidence_hypothetical_attributions": (
            result.low_confidence_hypothetical_attributions
        ),
        "low_confidence_actual_attributions": result.low_confidence_actual_attributions,
        "high_confidence_indices": evaluation.high_confidence_indices,
        "low_confidence_indices": evaluation.low_confidence_indices,
        "ineligible_indices": evaluation.ineligible_indices,
    }
    for name, array in arrays.items():
        arrays[name] = _save_array(out / f"{name}.npy", array)

    ids = list(sequence_ids)
    for label, indices in (
        ("high_confidence", evaluation.high_confidence_indices),
        ("low_confidence", evaluation.low_confidence_indices),
        ("ineligible", evaluation.ineligible_indices),
    ):
        selected = [ids[int(index)] for index in indices]
        _atomic_write_text(
            out / f"{label}_ids.txt",
            "".join(f"{identifier}\n" for identifier in selected),
        )

    diagnostic_path = out / "reference_diagnostics.csv"
    fields = tuple(ReferenceDiagnostic.__dataclass_fields__)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=out,
        prefix=".reference_diagnostics.",
        suffix=".tmp.csv",
        delete=False,
    ) as handle:
        temporary_diagnostics = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in evaluation.diagnostics)
    try:
        os.replace(temporary_diagnostics, diagnostic_path)
    finally:
        temporary_diagnostics.unlink(missing_ok=True)

    summary = {
        "analysis": "deep_shap",
        "attribution_method": "tangermeme.deep_lift_shap",
        "tangermeme_version": tangermeme_version,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "dataset": None if dataset is None else str(dataset),
        "model_target": result.target,
        "reference_config": reference_config.to_dict(),
        "confidence": {
            "threshold": evaluation.confidence_threshold,
            "minimum_mean_absolute_prediction_difference": evaluation.minimum_effect_size,
            "zero_atol": evaluation.zero_atol,
            "interpretation": (
                "directional consistency within the configured reference distribution; "
                "not statistically calibrated confidence"
            ),
        },
        "convergence": {
            "initial_local_ig_points": result.ig_points,
            "maximum_local_ig_points": result.max_ig_points,
            "atol": result.convergence_atol,
            "rtol": result.convergence_rtol,
            "unresolved_sequence_indices": result.unresolved_convergence_indices.tolist(),
        },
        "counts": {
            "selected": int(len(evaluation.sequence_indices)),
            "high_confidence": int(len(evaluation.high_confidence_indices)),
            "low_confidence": int(len(evaluation.low_confidence_indices)),
            "ineligible": int(len(evaluation.ineligible_indices)),
        },
        "reference_predictions_format": reference_predictions_format,
        "ordering": (
            "Attribution rows follow selected input order within each confidence "
            "partition and map to the corresponding *_indices.npy and *_ids.txt files."
        ),
        "outputs": {
            name: {"shape": list(np.asarray(array).shape), "dtype": str(np.asarray(array).dtype)}
            for name, array in arrays.items()
        },
    }
    _atomic_write_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")


def _contains_model(module: torch.nn.Module, model_type: type[torch.nn.Module]) -> bool:
    return any(isinstance(item, model_type) for item in module.modules())


def run_deep_shap(
    X: np.ndarray | torch.Tensor,
    predictor: Predictor | EnsemblePredictor,
    out_dir: str | Path,
    *,
    schema: str | SequenceSchema = "saluki6",
    sequence_ids: Sequence[str] | None = None,
    reference_config: ReferenceConfig | None = None,
    target: int = 0,
    prediction_batch_size: int | None = None,
    attribution_batch_size: int = 1,
    confidence_threshold: float = 0.95,
    minimum_effect_size: float | None = None,
    zero_atol: float = 1e-6,
    ig_points: int = 8,
    max_ig_points: int = 32,
    convergence_atol: float = 1e-3,
    convergence_rtol: float = 0.01,
    sequence_indices: Sequence[int] | None = None,
    sequence_start: int | None = None,
    sequence_end: int | None = None,
    sequence_shard_index: int | None = None,
    sequence_shards: int | None = None,
    reference_predictions_format: Literal["none", "csv", "parquet"] = "none",
    checkpoint: str | Path | None = None,
    dataset: str | Path | None = None,
) -> DeepSHAPAttributionResult:
    """Run reference generation, evaluation, attribution, and persistence."""

    arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    if arr.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {arr.shape}")
    ids = [str(i) for i in range(arr.shape[0])] if sequence_ids is None else list(sequence_ids)
    if len(ids) != arr.shape[0]:
        raise ValueError("sequence_ids length must match X.shape[0]")
    indices = resolve_analysis_indices(
        arr.shape[0],
        sequence_indices=sequence_indices,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        sequence_shard_index=sequence_shard_index,
        sequence_shards=sequence_shards,
    )
    config = (reference_config or ReferenceConfig()).normalized()
    generator = ReferenceGenerator(config, schema=schema)
    module, _, _, differentiable = _predictor_module(predictor)
    if not differentiable:
        raise TypeError("DeepSHAP requires differentiable PyTorch model predictors")
    if _contains_model(module, LegNet) and config.strategy not in {"full-mono", "full-dinuc"}:
        raise ValueError("MPRA-LegNet DeepSHAP supports only full-mono and full-dinuc references")
    target_predictor = _TargetPredictor(
        predictor,
        target=int(target),
        batch_size=prediction_batch_size,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prediction_path = None
    prediction_final_path = None
    if reference_predictions_format != "none":
        prediction_final_path = out / f"reference_predictions.{reference_predictions_format}"
        prediction_path = out / (
            f".reference_predictions.partial.{reference_predictions_format}"
        )
    evaluation = evaluate_references(
        arr,
        target_predictor,
        generator,
        sequence_ids=ids,
        sequence_indices=indices,
        batch_size=prediction_batch_size,
        confidence_threshold=confidence_threshold,
        minimum_effect_size=minimum_effect_size,
        zero_atol=zero_atol,
        reference_predictions_path=prediction_path,
        reference_predictions_format=reference_predictions_format,
    )
    result = compute_deep_shap_attributions(
        arr,
        predictor,
        generator,
        evaluation,
        target=int(target),
        attribution_batch_size=attribution_batch_size,
        ig_points=ig_points,
        max_ig_points=max_ig_points,
        convergence_atol=convergence_atol,
        convergence_rtol=convergence_rtol,
        storage_dir=out,
    )
    if prediction_path is not None and prediction_final_path is not None:
        os.replace(prediction_path, prediction_final_path)
    save_deep_shap_result(
        result,
        out,
        reference_config=config,
        sequence_ids=ids,
        checkpoint=checkpoint,
        dataset=dataset,
        reference_predictions_format=reference_predictions_format,
    )
    return result


def slice_attribution_inputs(
    dataset: str | Path,
    attribution_dir: str | Path,
    out_dir: str | Path,
    *,
    confidence: Literal["high", "low"] = "high",
    all_channels: bool = False,
) -> Path:
    """Write input sequences aligned to one saved confidence partition."""

    label = str(confidence).strip().lower()
    if label not in {"high", "low"}:
        raise ValueError("confidence must be 'high' or 'low'")
    partition = f"{label}_confidence"
    attribution_root = Path(attribution_dir)
    indices = np.load(attribution_root / f"{partition}_indices.npy")
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{partition}_indices.npy must be a one-dimensional integer array")
    bundle = load_bundle(dataset, mmap_mode="r")
    if np.any(indices < 0) or np.any(indices >= bundle.X.shape[0]):
        raise ValueError("Attribution indices fall outside the dataset")
    expected_ids = (attribution_root / f"{partition}_ids.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    selected_ids = [str(bundle.ids[int(index)]) for index in indices]
    if selected_ids != expected_ids:
        raise ValueError("Attribution IDs do not match the dataset at the saved indices")
    expected_rows = np.load(
        attribution_root / f"{partition}_hypothetical_attributions.npy", mmap_mode="r"
    ).shape[0]
    if expected_rows != len(indices):
        raise ValueError("Attribution tensor and saved indices have different row counts")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    channels = bundle.X.shape[1] if all_channels else 4
    np.save(out / "sequences.npy", np.asarray(bundle.X[indices, :channels]))
    np.save(out / "sequence_indices.npy", indices.astype(np.int64, copy=False))
    (out / "ids.txt").write_text(
        "".join(f"{identifier}\n" for identifier in selected_ids), encoding="utf-8"
    )
    if bundle.metadata is not None:
        selected_metadata = [bundle.metadata[int(index)] for index in indices]
        (out / "metadata.json").write_text(
            json.dumps(selected_metadata, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "analysis": "slice_attribution_inputs",
        "dataset": str(dataset),
        "attribution_dir": str(attribution_dir),
        "confidence": label,
        "all_channels": bool(all_channels),
        "shape": [int(len(indices)), int(channels), int(bundle.X.shape[-1])],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out
