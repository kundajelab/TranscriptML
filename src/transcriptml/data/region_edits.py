from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from transcriptml.data.schemas import SequenceSchema, get_schema


def base_channel_indices(schema: str | SequenceSchema = "saluki6") -> np.ndarray:
    """Resolve exactly one A, C, G, and U/T channel."""

    resolved = get_schema(schema)
    indices = []
    letters = []
    for base_name in resolved.base_channels:
        if base_name not in resolved.channels:
            raise ValueError(
                f"Base channel {base_name!r} is not present in schema channels {resolved.channels}"
            )
        letter = base_name.upper().replace("T", "U")
        if letter not in {"A", "C", "G", "U"}:
            raise ValueError(f"Unsupported base channel {base_name!r}; expected A/C/G/U/T")
        indices.append(resolved.channels.index(base_name))
        letters.append(letter)
    if set(letters) != {"A", "C", "G", "U"} or len(letters) != 4:
        raise ValueError("region edits require exactly one A, C, G, and U/T base channel")
    return np.asarray(indices, dtype=np.int64)


def resolve_cds_channel(
    schema: str | SequenceSchema = "saluki6",
    cds_channel: str | int | None = None,
) -> int:
    """Resolve a CDS channel name/index using TranscriptML conventions."""

    resolved = get_schema(schema)
    if isinstance(cds_channel, int):
        if cds_channel < 0 or cds_channel >= resolved.n_channels:
            raise ValueError(
                f"cds_channel index {cds_channel} is outside schema with "
                f"{resolved.n_channels} channels"
            )
        return int(cds_channel)
    if isinstance(cds_channel, str):
        try:
            return resolved.channels.index(cds_channel)
        except ValueError as exc:
            raise ValueError(
                f"cds_channel {cds_channel!r} is not in schema channels {resolved.channels}"
            ) from exc

    preferred = ("CDS_codon_start", "cds_codon_start", "codon_start", "CDS", "cds")
    lower_to_index = {name.lower(): i for i, name in enumerate(resolved.channels)}
    for name in preferred:
        if name.lower() in lower_to_index:
            return lower_to_index[name.lower()]
    for i, name in enumerate(resolved.channels):
        lowered = name.lower()
        if "cds" in lowered or "coding" in lowered or "codon_start" in lowered:
            return i
    raise ValueError("Could not infer CDS channel from schema; pass cds_channel explicitly")


def valid_length_from_bases(x: np.ndarray, base_channels: np.ndarray) -> int:
    """Infer represented length from nonzero base-channel columns."""

    base = np.asarray(x[np.asarray(base_channels, dtype=np.int64)])
    idx = np.nonzero(np.any(base != 0, axis=0))[0]
    return int(idx[-1] + 1) if idx.size else 0


def base_symbols(x: np.ndarray, start: int, end: int, base_channels: np.ndarray) -> np.ndarray:
    """Read base-channel offsets, retaining ambiguous positions as ``-1``."""

    if end <= start:
        return np.empty((0,), dtype=np.int16)
    channels = np.asarray(base_channels, dtype=np.int64)
    region = np.asarray(x[channels, int(start) : int(end)])
    called = np.count_nonzero(region, axis=0) == 1
    symbols = np.full(region.shape[1], -1, dtype=np.int16)
    if np.any(called):
        symbols[called] = np.argmax(region[:, called], axis=0).astype(np.int16, copy=False)
    return symbols


def write_base_symbols(
    x: np.ndarray,
    start: int,
    end: int,
    symbols: np.ndarray,
    base_channels: np.ndarray,
) -> None:
    """Write base-channel offsets while leaving annotations untouched."""

    if end <= start:
        return
    start = int(start)
    end = int(end)
    channels = np.asarray(base_channels, dtype=np.int64)
    x[channels, start:end] = 0
    valid = np.asarray(symbols) >= 0
    if not np.any(valid):
        return
    columns = start + np.nonzero(valid)[0]
    offsets = np.asarray(symbols[valid], dtype=np.int64)
    x[channels[offsets], columns] = 1


def region_bounds(
    region: str,
    *,
    valid_length: int,
    cds: Any | None,
) -> tuple[int, int] | None:
    """Return transcript, UTR, or CDS half-open coordinates."""

    if region == "transcript":
        return 0, int(valid_length)
    if cds is None or int(cds.cds_length) < 3 or np.asarray(cds.starts).size == 0:
        return None
    cds_start = max(0, int(cds.cds_start))
    cds_end = min(int(valid_length), int(cds.cds_end) + 1)
    if region == "5utr":
        return 0, cds_start
    if region == "cds":
        return cds_start, cds_end
    if region == "3utr":
        return cds_end, int(valid_length)
    raise ValueError("region must be one of: 5utr, cds, 3utr, transcript")


def shuffle_nucleotides_inplace(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    base_channels: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Permute nucleotide/ambiguous symbols within one region."""

    symbols = base_symbols(x, start, end, base_channels)
    if symbols.size > 1:
        symbols = symbols[rng.permutation(symbols.size)]
    write_base_symbols(x, start, end, symbols, base_channels)


def randomize_nucleotides_inplace(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    base_channels: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Replace every position in one region with an IID A/C/G/U base."""

    length = max(0, int(end) - int(start))
    channels = np.asarray(base_channels, dtype=np.int64)
    symbols = rng.integers(0, int(channels.size), size=length, dtype=np.int16)
    write_base_symbols(x, start, end, symbols, channels)


def shuffle_codons_inplace(
    x: np.ndarray,
    *,
    cds: Any,
    base_channels: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Permute complete annotated CDS codons as three-base units."""

    starts = np.asarray(cds.starts, dtype=np.int64)
    starts = starts[(starts >= int(cds.cds_start)) & (starts + 2 <= int(cds.cds_end))]
    if starts.size == 0:
        return
    codons = np.stack(
        [base_symbols(x, int(start), int(start) + 3, base_channels) for start in starts],
        axis=0,
    )
    if codons.shape[0] > 1:
        codons = codons[rng.permutation(codons.shape[0])]
    for start, codon in zip(starts.tolist(), codons, strict=True):
        write_base_symbols(x, int(start), int(start) + 3, codon, base_channels)


def randomize_synonymous_codons_inplace(
    x: np.ndarray,
    *,
    cds: Any,
    base_channels: np.ndarray,
    base_letters: Sequence[str],
    synonymous_alternates: Mapping[str, Sequence[str]],
    rng: np.random.Generator,
) -> int:
    """Replace each decodable CDS codon with a random synonymous alternate.

    Codons absent from ``synonymous_alternates`` or mapped to an empty sequence
    are left unchanged. This lets callers explicitly preserve stop codons and
    single-codon amino acids. Ambiguous codons are also retained verbatim.

    Returns:
        Number of codons replaced.
    """

    channels = np.asarray(base_channels, dtype=np.int64)
    letters = tuple(str(letter).upper().replace("T", "U") for letter in base_letters)
    if channels.size != 4 or len(letters) != 4 or set(letters) != {"A", "C", "G", "U"}:
        raise ValueError("synonymous codon edits require aligned A/C/G/U base channels")
    base_to_offset = {base: offset for offset, base in enumerate(letters)}

    starts = np.asarray(cds.starts, dtype=np.int64)
    starts = starts[(starts >= int(cds.cds_start)) & (starts + 2 <= int(cds.cds_end))]
    n_replaced = 0
    for start in starts.tolist():
        symbols = base_symbols(x, int(start), int(start) + 3, channels)
        if symbols.size != 3 or np.any(symbols < 0):
            continue
        reference = "".join(letters[int(offset)] for offset in symbols)
        alternates = tuple(
            codon.upper().replace("T", "U")
            for codon in synonymous_alternates.get(reference, ())
            if codon.upper().replace("T", "U") != reference
        )
        if not alternates:
            continue
        alternate = alternates[int(rng.integers(0, len(alternates)))]
        try:
            replacement = np.asarray([base_to_offset[base] for base in alternate], dtype=np.int16)
        except KeyError as exc:
            raise ValueError(f"Invalid synonymous codon {alternate!r}") from exc
        if replacement.shape != (3,):
            raise ValueError(f"Invalid synonymous codon {alternate!r}")
        write_base_symbols(x, int(start), int(start) + 3, replacement, channels)
        n_replaced += 1
    return n_replaced


def shuffle_nucleotides_batch(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    base_channels: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``n_replicates`` copies with ``[start, end)`` base symbols permuted."""

    arr = np.asarray(x)
    count = int(n_replicates)
    mutants = np.repeat(arr[None], count, axis=0)
    if count <= 0 or int(end) <= int(start):
        return mutants
    s, e = int(start), int(end)
    channels = np.asarray(base_channels, dtype=np.int64)
    region = np.asarray(arr[channels, s:e])
    called = np.count_nonzero(region, axis=0) == 1
    clean = np.zeros_like(region)
    if np.any(called):
        clean[np.argmax(region[:, called], axis=0), called] = 1
    reg_len = clean.shape[1]
    if reg_len > 1:
        perms = np.stack([rng.permutation(reg_len) for _ in range(count)], axis=0)
        edited = np.transpose(clean[:, perms], (1, 0, 2))
    else:
        edited = np.broadcast_to(clean[None], (count, clean.shape[0], reg_len))
    if channels.shape == (4,) and np.array_equal(channels, np.arange(4, dtype=np.int64)):
        mutants[:, :4, s:e] = edited
    else:
        mutants[:, channels, s:e] = edited
    return mutants


def randomize_nucleotides_batch(
    x: np.ndarray,
    *,
    start: int,
    end: int,
    base_channels: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``n_replicates`` copies with ``[start, end)`` replaced by IID bases."""

    arr = np.asarray(x)
    count = int(n_replicates)
    mutants = np.repeat(arr[None], count, axis=0)
    s, e = int(start), int(end)
    reg_len = max(0, e - s)
    if count <= 0 or reg_len == 0:
        return mutants
    channels = np.asarray(base_channels, dtype=np.int64)
    n_ch = int(channels.size)
    choices = rng.integers(0, n_ch, size=(count, reg_len), dtype=np.int8)
    edited = (choices[:, None, :] == np.arange(n_ch, dtype=np.int8)[None, :, None]).astype(
        arr.dtype, copy=False
    )
    if channels.shape == (4,) and np.array_equal(channels, np.arange(4, dtype=np.int64)):
        mutants[:, :4, s:e] = edited
    else:
        mutants[:, channels, s:e] = edited
    return mutants


def shuffle_codons_batch(
    x: np.ndarray,
    *,
    cds: Any,
    base_channels: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``n_replicates`` copies with complete annotated CDS codons permuted."""

    arr = np.asarray(x)
    count = int(n_replicates)
    mutants = np.repeat(arr[None], count, axis=0)
    if count <= 0:
        return mutants
    starts = np.asarray(cds.starts, dtype=np.int64)
    starts = starts[(starts >= int(cds.cds_start)) & (starts + 2 <= int(cds.cds_end))]
    if starts.size == 0:
        return mutants
    channels = np.asarray(base_channels, dtype=np.int64)
    k = int(starts.size)
    pos3 = starts[:, None] + np.arange(3, dtype=np.int64)[None, :]
    raw = np.asarray(arr[channels[:, None, None], pos3[None, :, :]])
    called = np.count_nonzero(raw, axis=0) == 1
    clean = np.zeros_like(raw)
    if np.any(called):
        k_idx, b_idx = np.nonzero(called)
        clean[np.argmax(raw[:, k_idx, b_idx], axis=0), k_idx, b_idx] = 1
    codon_ohe = np.transpose(clean, (1, 0, 2))
    if k > 1:
        perms = np.stack([rng.permutation(k) for _ in range(count)], axis=0)
        shuffled = codon_ohe[perms]
    else:
        shuffled = np.repeat(codon_ohe[None], count, axis=0)
    flat_ohe = np.transpose(shuffled, (0, 2, 1, 3)).reshape(count, channels.size, 3 * k)

    s0 = int(starts[0])
    is_contiguous_codons = (
        int(starts[-1]) - s0 == 3 * (k - 1)
        and np.array_equal(starts, np.arange(s0, s0 + 3 * k, 3, dtype=np.int64))
    )
    if is_contiguous_codons:
        if channels.shape == (4,) and np.array_equal(channels, np.arange(4, dtype=np.int64)):
            mutants[:, :4, s0 : s0 + 3 * k] = flat_ohe
        else:
            mutants[:, channels, s0 : s0 + 3 * k] = flat_ohe
    else:
        pos_flat = pos3.reshape(-1)
        mutants[:, channels[:, None], pos_flat[None, :]] = flat_ohe
    return mutants


def _build_synonymous_codon_tables(
    letters: tuple[str, ...],
    synonymous_alternates: Mapping[str, Sequence[str]],
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build compact 64-codon lookup tables for vectorized synonymous sampling."""

    base_to_offset = {base: offset for offset, base in enumerate(letters)}
    n_alts = np.zeros(64, dtype=np.int64)
    alt_ids = np.zeros((64, 64), dtype=np.int64)
    id_to_ohe = np.zeros((64, 4, 3), dtype=dtype)
    for b0 in range(4):
        for b1 in range(4):
            for b2 in range(4):
                cid = b0 * 16 + b1 * 4 + b2
                id_to_ohe[cid, b0, 0] = 1
                id_to_ohe[cid, b1, 1] = 1
                id_to_ohe[cid, b2, 2] = 1
                reference = f"{letters[b0]}{letters[b1]}{letters[b2]}"
                raw_alts = synonymous_alternates.get(reference, ())
                valid_alt_ids: list[int] = []
                for raw_codon in raw_alts:
                    alt = str(raw_codon).upper().replace("T", "U")
                    if alt == reference:
                        continue
                    if len(alt) != 3:
                        raise ValueError(f"Invalid synonymous codon {alt!r}")
                    try:
                        a0 = base_to_offset[alt[0]]
                        a1 = base_to_offset[alt[1]]
                        a2 = base_to_offset[alt[2]]
                    except KeyError as exc:
                        raise ValueError(f"Invalid synonymous codon {alt!r}") from exc
                    valid_alt_ids.append(a0 * 16 + a1 * 4 + a2)
                n_alts[cid] = len(valid_alt_ids)
                if valid_alt_ids:
                    alt_ids[cid, : len(valid_alt_ids)] = valid_alt_ids
    return n_alts, alt_ids, id_to_ohe


def randomize_synonymous_codons_batch(
    x: np.ndarray,
    *,
    cds: Any,
    base_channels: np.ndarray,
    base_letters: Sequence[str],
    synonymous_alternates: Mapping[str, Sequence[str]],
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``n_replicates`` copies with decodable CDS codons synonymously recoded."""

    channels = np.asarray(base_channels, dtype=np.int64)
    letters = tuple(str(letter).upper().replace("T", "U") for letter in base_letters)
    if channels.size != 4 or len(letters) != 4 or set(letters) != {"A", "C", "G", "U"}:
        raise ValueError("synonymous codon edits require aligned A/C/G/U base channels")

    arr = np.asarray(x)
    count = int(n_replicates)
    mutants = np.repeat(arr[None], count, axis=0)
    if count <= 0:
        return mutants

    starts = np.asarray(cds.starts, dtype=np.int64)
    starts = starts[(starts >= int(cds.cds_start)) & (starts + 2 <= int(cds.cds_end))]
    if starts.size == 0:
        return mutants

    n_alts, alt_ids, id_to_ohe = _build_synonymous_codon_tables(
        letters, synonymous_alternates, arr.dtype
    )
    pos3 = starts[:, None] + np.arange(3, dtype=np.int64)[None, :]
    raw = np.asarray(arr[channels[:, None, None], pos3[None, :, :]])
    called = np.count_nonzero(raw, axis=0) == 1
    valid_codon = np.all(called, axis=1)
    if not np.any(valid_codon):
        return mutants

    valid_idx = np.flatnonzero(valid_codon)
    offsets = np.argmax(raw[:, valid_idx, :], axis=0)
    codon_ids = offsets[:, 0] * 16 + offsets[:, 1] * 4 + offsets[:, 2]
    has_alt = n_alts[codon_ids] > 0
    if not np.any(has_alt):
        return mutants

    mutable_idx = valid_idx[has_alt]
    mutable_ids = codon_ids[has_alt]
    mutable_counts = n_alts[mutable_ids]
    m = int(mutable_idx.size)
    choices = rng.integers(0, mutable_counts[None, :], size=(count, m))
    new_ids = alt_ids[mutable_ids[None, :], choices]
    new_ohe = id_to_ohe[new_ids]
    flat_ohe = np.transpose(new_ohe, (0, 2, 1, 3)).reshape(count, 4, 3 * m)
    m_pos = pos3[mutable_idx].reshape(-1)
    mutants[:, channels[:, None], m_pos[None, :]] = flat_ohe
    return mutants
