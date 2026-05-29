from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from transcriptml.data.schemas import RNA4, SALUKI6, SequenceSchema, get_schema

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
INDEX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "U"}
DEFAULT_SALUKI_LENGTH = 12_288
_BASE_LUT = np.full(256, -1, dtype=np.int16)
for _base, _idx in BASE_TO_INDEX.items():
    _BASE_LUT[ord(_base)] = _idx
    _BASE_LUT[ord(_base.lower())] = _idx


def fixed_length_sequence(seq: str, length: int, *, truncate_from: str = "5prime") -> tuple[str, int]:
    """Right-pad short sequences and truncate long sequences.

    ``truncate_from="5prime"`` preserves the 3-prime-most bases, matching the
    Saluki-style legacy pipeline. The returned offset is the number of original
    bases removed from the 5-prime side.

    Args:
        seq: Input nucleotide sequence. Values are coerced to ``str``.
        length: Positive fixed output length.
        truncate_from: Side to truncate when ``seq`` is longer than ``length``;
            accepts ``"5prime"``/``"left"`` or ``"3prime"``/``"right"``.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    seq = str(seq)
    if len(seq) < length:
        return seq + ("N" * (length - len(seq))), 0
    if len(seq) == length:
        return seq, 0
    if truncate_from in {"5prime", "left"}:
        offset = len(seq) - length
        return seq[-length:], offset
    if truncate_from in {"3prime", "right"}:
        return seq[:length], 0
    raise ValueError("truncate_from must be '5prime'/'left' or '3prime'/'right'")


def encode_rna_sequence(
    seq: str,
    *,
    length: int | None = None,
    dtype: np.dtype | type = np.uint8,
    truncate_from: str = "5prime",
) -> np.ndarray:
    """Encode RNA sequence as ``(4, L)`` A/C/G/U one-hot.

    T is treated as U. N and all unknown symbols are encoded as all-zero columns.

    Args:
        seq: Input RNA or DNA sequence string.
        length: Optional fixed length. When provided, the sequence is padded or
            truncated before encoding.
        dtype: NumPy dtype for the returned one-hot array.
        truncate_from: Side to truncate when ``length`` is provided and ``seq``
            is too long.
    """

    if length is not None:
        seq, _ = fixed_length_sequence(seq, length, truncate_from=truncate_from)
    seq = str(seq)
    L = len(seq)
    out = np.zeros((4, L), dtype=dtype)
    if L == 0:
        return out
    try:
        encoded = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    except UnicodeEncodeError:
        encoded = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    idx = _BASE_LUT[encoded]
    valid = idx >= 0
    if np.any(valid):
        pos = np.nonzero(valid)[0]
        out[idx[valid].astype(np.intp, copy=False), pos] = 1
    return out


def encode_sequences(
    seqs: Sequence[str],
    *,
    length: int | None = None,
    dtype: np.dtype | type = np.uint8,
    truncate_from: str = "5prime",
) -> np.ndarray:
    """Encode a collection of RNA sequences as ``(N, 4, L)``.

    Args:
        seqs: Sequence of RNA or DNA sequence strings to encode.
        length: Optional fixed length for every encoded sequence. When omitted,
            the maximum input sequence length is used.
        dtype: NumPy dtype for the returned one-hot array.
        truncate_from: Side to truncate when fixed-length encoding shortens a
            sequence.
    """

    seqs = list(seqs)
    if length is None:
        length = max((len(str(s)) for s in seqs), default=0)
    if length <= 0:
        raise ValueError("Cannot encode an empty collection without a positive length")
    X = np.zeros((len(seqs), 4, length), dtype=dtype)
    for i, seq in enumerate(seqs):
        X[i] = encode_rna_sequence(seq, length=length, dtype=dtype, truncate_from=truncate_from)
    return X


def _adjust_positions(
    positions: Iterable[int] | None,
    *,
    offset: int,
    length: int,
    original_length: int,
) -> list[int]:
    """Shift original transcript positions into a truncated fixed-length window.

    Args:
        positions: Optional iterable of original transcript-coordinate
            positions.
        offset: Number of original bases removed from the 5-prime side.
        length: Fixed encoded sequence length.
        original_length: Length of the original untruncated transcript.
    """

    if positions is None:
        return []
    adjusted: list[int] = []
    for pos in positions:
        original_pos = int(pos)
        if original_pos < 0 or original_pos >= original_length:
            continue
        p = original_pos - offset
        if 0 <= p < length:
            adjusted.append(p)
    return adjusted


def encode_saluki_transcript(
    seq: str,
    *,
    length: int = DEFAULT_SALUKI_LENGTH,
    cds_positions: Iterable[int] | None = None,
    splice_positions: Iterable[int] | None = None,
    dtype: np.dtype | type = np.uint8,
) -> np.ndarray:
    """Encode a transcript as Saluki-style ``(6, L)``.

    Short transcripts are right-padded with N/all-zero columns. Long transcripts
    are truncated from the 5-prime side so the represented window is the
    3-prime-most ``length`` bases. Annotation positions are expected in original
    transcript coordinates and are shifted by the same truncation offset.

    Args:
        seq: Input transcript sequence.
        length: Fixed Saluki input length.
        cds_positions: Optional CDS annotation positions in original transcript
            coordinates.
        splice_positions: Optional splice annotation positions in original
            transcript coordinates.
        dtype: NumPy dtype for the returned encoded array.
    """

    original_len = len(str(seq))
    fixed_seq, offset = fixed_length_sequence(seq, length, truncate_from="5prime")
    out = np.zeros((6, length), dtype=dtype)
    out[:4] = encode_rna_sequence(fixed_seq, dtype=dtype)
    for pos in _adjust_positions(cds_positions, offset=offset, length=length, original_length=original_len):
        out[4, pos] = 1
    for pos in _adjust_positions(splice_positions, offset=offset, length=length, original_length=original_len):
        out[5, pos] = 1
    return out


def infer_valid_length(x: np.ndarray, *, base_channels: int | None = None) -> int:
    """Infer the last represented non-zero column plus one.

    This follows the legacy convention that N-padding is contiguous all-zero
    padding at the right edge. Unknown all-zero bases inside a valid transcript
    are allowed; they do not by themselves end the sequence.

    Args:
        x: Encoded ``(C, L)`` sequence array.
        base_channels: Optional number of leading channels to inspect. When
            ``None``, all channels are considered.
    """

    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a (C, L) array, got shape {arr.shape}")
    use = arr[:base_channels] if base_channels is not None else arr
    nonzero = np.any(use != 0, axis=0)
    idx = np.nonzero(nonzero)[0]
    return int(idx[-1] + 1) if idx.size else 0


def infer_valid_lengths(X: np.ndarray, *, base_channels: int | None = None) -> np.ndarray:
    """Infer valid sequence lengths for a batch of encoded arrays.

    Args:
        X: Encoded ``(N, C, L)`` batch array.
        base_channels: Optional number of leading channels to inspect for each
            sequence. When ``None``, all channels are considered.
    """

    arr = np.asarray(X)
    if arr.ndim != 3:
        raise ValueError(f"Expected a (N, C, L) array, got shape {arr.shape}")
    return np.array([infer_valid_length(x, base_channels=base_channels) for x in arr], dtype=np.int64)


def decode_rna_one_hot(x: np.ndarray, *, unknown: str = "N") -> str:
    """Decode base channels without treating all-zero columns as A.

    Args:
        x: Encoded array with at least four leading base channels and shape
            ``(C, L)``.
        unknown: Character to emit for ambiguous or all-zero columns.
    """

    arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[0] < 4:
        raise ValueError(f"Expected at least four base channels, got shape {arr.shape}")
    chars: list[str] = []
    for pos in range(arr.shape[1]):
        col = arr[:4, pos]
        if np.count_nonzero(col) != 1:
            chars.append(unknown)
        else:
            chars.append(INDEX_TO_BASE[int(np.argmax(col))])
    return "".join(chars)
