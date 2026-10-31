r"""Read Tecplot ASCII DAT files.

Supported ``DATAPACKING`` modes:
    * ``BLOCK``, one contiguous value block per variable (Tecplot default).
    * ``POINT``, one row of all variable values per node, followed by a separate
      row-per-cell section for any cell-centered variables. This is the format most
      commonly produced by third-party exporters and tools that treat the file like a
      CSV with a header.

The entire file is parsed into memory at construction, no lazy disk reads, unlike SZL
and (for data values) PLT.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ._constants import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
)
from ._containers import ZoneList
from ._reader import (
    PolyArrayKey,
    TecplotAuxDataReader,
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotPolyZoneReader,
    TecplotReader,
    TecplotVariableReader,
    TecplotZoneReader,
    convert_face_connections_to_zero_based,
)

# --------------------------------------------------------------------------------------
# Shared module-level constants
# --------------------------------------------------------------------------------------

# FE zone types fully supported for reading and writing
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})

# Zone types whose connectivity is face-based
_FE_POLY: frozenset[ZoneType] = frozenset({
    ZoneType.FEPOLYGON,
    ZoneType.FEPOLYHEDRON,
})

# Nodes per element for each supported simple FE type
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

# ASCII keyword to ZoneType
_STR_TO_ZONETYPE: dict[str, ZoneType] = {
    "ordered": ZoneType.ORDERED,
    "felineseg": ZoneType.FELINESEG,
    "fetriangle": ZoneType.FETRIANGLE,
    "fequadrilateral": ZoneType.FEQUADRILATERAL,
    "fetetrahedron": ZoneType.FETETRAHEDRON,
    "febrick": ZoneType.FEBRICK,
    "fepolygon": ZoneType.FEPOLYGON,
    "fepolyhedron": ZoneType.FEPOLYHEDRON,
}

# ASCII keyword to FileType
_STR_TO_FILETYPE: dict[str, FileType] = {
    "full": FileType.FULL,
    "grid": FileType.GRID,
    "solution": FileType.SOLUTION,
}

# ASCII keyword to DataPacking
_STR_TO_DATAPACKING: dict[str, DataPacking] = {
    "point": DataPacking.POINT,
    "block": DataPacking.BLOCK,
}

# ASCII keyword to FaceNeighborMode
_STR_TO_FACENEIGHBORMODE: dict[str, FaceNeighborMode] = {
    "localonetoone": FaceNeighborMode.LOCAL_ONE_TO_ONE,
    "localonetomany": FaceNeighborMode.LOCAL_ONE_TO_MANY,
    "globalonetoone": FaceNeighborMode.GLOBAL_ONE_TO_ONE,
    "globalonetomany": FaceNeighborMode.GLOBAL_ONE_TO_MANY,
}

# Legacy F= (format) keyword to (is_fe, DataPacking)
_STR_TO_LEGACY_FORMAT: dict[str, tuple[bool, DataPacking]] = {
    "POINT": (False, DataPacking.POINT),
    "BLOCK": (False, DataPacking.BLOCK),
    "FEPOINT": (True, DataPacking.POINT),
    "FEBLOCK": (True, DataPacking.BLOCK),
}

# Legacy ``ET=`` (element type) keyword to ZoneType
_STR_TO_ELEMENT_TYPE: dict[str, ZoneType] = {
    "LINESEG": ZoneType.FELINESEG,
    "TRIANGLE": ZoneType.FETRIANGLE,
    "QUADRILATERAL": ZoneType.FEQUADRILATERAL,
    "TETRAHEDRON": ZoneType.FETETRAHEDRON,
    "BRICK": ZoneType.FEBRICK,
}

# ZoneType to ASCII keyword
_ZONETYPE_STR: dict[ZoneType, str] = {
    ZoneType.ORDERED: "Ordered",
    ZoneType.FELINESEG: "FELineSeg",
    ZoneType.FETRIANGLE: "FETriangle",
    ZoneType.FEQUADRILATERAL: "FEQuadrilateral",
    ZoneType.FETETRAHEDRON: "FETetrahedron",
    ZoneType.FEBRICK: "FEBrick",
    ZoneType.FEPOLYGON: "FEPolygon",
    ZoneType.FEPOLYHEDRON: "FEPolyhedron",
}

# FileType to ASCII keyword
_FILETYPE_STR: dict[FileType, str] = {
    FileType.GRID: "GRID",
    FileType.SOLUTION: "SOLUTION",
}

# DataType to NumPy dtype string
_DT_TO_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}

# ASCII DT= keyword (and common aliases) to DataType
_STR_TO_DATATYPE: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "longint": DataType.INT32,
    "shortint": DataType.INT16,
    "byte": DataType.BYTE,
}

#: Values per line for Write data blocks.
_VALUES_PER_LINE: int = 5

# Safety cap on the number of grow-and-reparse iterations the vectorized block readers
# will attempt before giving up and falling back to the tolerant token-by-token parser
_MAX_FAST_BLOCK_ITERATIONS: int = 64


# ======================================================================================
# Shared internal helpers
# ======================================================================================


def _quote(s: str) -> str:
    """Wrap *s* in double-quotes, escaping embedded double-quotes.

    Example:
        >>> q = _quote("hello world")
    """
    return '"' + str(s).replace('"', '\\"') + '"'


def _unquote(s: str) -> str:
    r"""Remove surrounding double-quotes and unescape internal ``\\"``.

    Example:
        >>> s = _unquote('"hello \\"world\\""')
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\"', '"')
    return s


def _strip_comment(line: str) -> str:
    """Remove a Tecplot ``#`` comment and trailing whitespace.

    Example:
        >>> clean = _strip_comment(line)
    """
    idx = line.find("#")
    return line[:idx].rstrip() if idx >= 0 else line.rstrip()


def _is_float_block_boundary(line: str) -> bool:
    r"""Return ``True`` if a stripped *line* ends a float data block early.

    Shared by :meth:`TecplotDatReader._read_float_block_slow` (line-by-line) and
    :meth:`TecplotDatReader._read_float_block_fast` (bulk, on parse failure) so the two
    readers agree on exactly where a block stops.

    Example:
        >>> _is_float_block_boundary("ZONE T=\\"next\\"")
        True
    """
    upper = line.lstrip().upper()
    if not upper:
        return False
    if upper.split("=")[0].split()[0] == "ZONE":
        return True
    return (
        upper.startswith(("DATASETAUXDATA", "VARAUXDATA"))
        or upper.startswith("TITLE")
        and "=" in upper
        or upper.startswith("VARIABLES")
        and "=" in upper
    )


def _is_int_block_boundary(line: str) -> bool:
    r"""Return ``True`` if a stripped *line* ends an int (connectivity) block.

    Example:
        >>> _is_int_block_boundary("DATASETAUXDATA foo=\\"bar\\"")
        True
    """
    upper = line.lstrip().upper()
    return upper.startswith(("ZONE", "DATASETAUXDATA"))


def _infer_data_type(arr: npt.NDArray) -> DataType:
    """Return the most appropriate :class:`DataType` for *arr*'s dtype.

    Example:
        >>> dt = _infer_data_type(arr)
    """
    dtype = arr.dtype
    if dtype.kind == "f":
        return DataType.DOUBLE if dtype.itemsize >= 8 else DataType.FLOAT
    if dtype.kind in ("i", "u"):
        if dtype.itemsize >= 4:
            return DataType.INT32
        if dtype.itemsize == 2:
            return DataType.INT16
        return DataType.BYTE
    return DataType.FLOAT


# ======================================================================================
# Parsing helpers (used only by Read)
# ======================================================================================


def _extract_quoted_strings(text: str) -> list[str]:
    """Return all double-quoted strings found in *text* (content unescaped).

    Example:
        >>> names = _extract_quoted_strings('"X" "Y" "Pressure"')
    """
    return [
        m.group(1).replace('\\"', '"')
        for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', text)
    ]


def _next_token_is_value(text: str, i: int) -> bool:
    """Return ``True`` if the token at *i* is a value, not the next key.

    Supports the legacy whitespace-separated ``KEY VALUE`` form in zone headers
    (e.g. ``STRANDID 2``). A token is treated as the *next key* — and therefore not a
    value for the preceding key — when it is immediately followed (after optional
    whitespace) by ``=``. A quoted token is always a value.

    Example:
        >>> _next_token_is_value("STRANDID 2", 9)
        True

    Args:
        text: The full header string being parsed.
        i: Index of the first character of the candidate token.

    Returns:
        ``True`` if the token should be consumed as the current key's value.
    """
    n = len(text)
    if i < n and text[i] == '"':
        return True
    j = i
    while j < n and text[j] not in " \t,=":
        j += 1
    # Skip trailing whitespace and see whether the token is followed by '='.
    while j < n and text[j] in " \t":
        j += 1
    return not (j < n and text[j] == "=")


def _determine_zone_type(kv: dict[str, str]) -> ZoneType:
    """Determine a zone's type from its header key-value pairs.

    Shared by the full parser and the metadata-only scanner so both agree on the same
    modern/legacy keyword rules.

    Two header dialects are supported:
        * Modern: ``ZONETYPE=FEQuadrilateral, DATAPACKING=POINT``
        * Legacy: ``F=FEPOINT, ET=QUADRILATERAL``

    The FE-vs-ordered distinction comes from ``ZONETYPE`` (modern) or ``F``
    (legacy). ``ET`` (element type) only ever appears on finite-element zones, so it
    merely names the element *shape* once a zone is already known to be FE. Modern
    keywords win when present.

    Args:
        kv: Zone header key-value pairs (leading ``ZONE`` keyword already stripped), as
            returned by :func:`_kv_split`.

    Returns:
        The zone's type. Unlike the full parser, does not reject FEPOLYGON/FEPOLYHEDRON,
        those just aren't readable for data, the header itself is perfectly legible.

    Raises:
        ValueError: If a legacy FE header (``F=FEPOINT``/``FEBLOCK``) is
            missing the required ``ET`` keyword.

    Example:
        >>> _determine_zone_type({"T": '"z"', "I": "5"})
        <ZoneType.ORDERED: ...>
    """
    legacy_is_fe: bool | None = None
    if "F" in kv:
        legacy_is_fe, _ = _parse_legacy_format(kv["F"])

    if "ZONETYPE" in kv:
        zt_raw = kv["ZONETYPE"].rstrip(",").strip().lower()
        return _STR_TO_ZONETYPE.get(zt_raw, ZoneType.ORDERED)
    elif legacy_is_fe is False:
        return ZoneType.ORDERED
    elif legacy_is_fe or "ET" in kv:
        if "ET" not in kv:
            raise ValueError(
                "Legacy FE zone header specifies F=FEPOINT/FEBLOCK but is "
                "missing the required ET (element type) keyword."
            )
        return _parse_legacy_element_type(kv["ET"])
    else:
        return ZoneType.ORDERED


def _collect_zone_header_lines(tokens: _LineBuffer) -> list[str]:
    """Collect a ZONE block's header lines, stopping before data begins.

    Shared by the full parser and the metadata-only scanner (:func:`peek_dat_metadata`)
    so both agree on exactly where a zone's header ends and its data starts.

    Args:
        tokens: Positioned at the ``ZONE`` line itself (not yet consumed).

    Returns:
        Raw header lines, including the leading ``ZONE`` keyword line.

    Example:
        >>> lines = _collect_zone_header_lines(tokens)
    """
    header_lines: list[str] = [tokens.next_stripped()]  # ZONE T=...
    while tokens.has_more():
        nxt = tokens.peek_stripped()
        if not nxt:
            tokens.next_stripped()
            continue
        nxt_upper = nxt.lstrip().upper()
        # Stop at a new zone, top-level keyword, or the first data line.
        if nxt_upper.split("=")[0].split()[0] == "ZONE":
            break
        if nxt_upper.startswith(("DATASETAUXDATA", "VARAUXDATA")):
            break
        first_ch = nxt.lstrip()[0] if nxt.lstrip() else ""
        # A data line begins with a numeric token (covers leading-dot values such as
        # ``.5`` and signed values such as ``-1.2e3``).
        if first_ch in "0123456789+-.":
            break
        header_lines.append(tokens.next_stripped())
    return header_lines


def _kv_split(text: str) -> dict[str, str]:
    """Parse a loose ``KEY=VALUE`` string into an upper-cased-key dict.

    Handles quoted values, parenthesised blocks (VARLOCATION, VARSHARELIST), and
    bracketed lists (PASSIVEVARLIST).

    Example:
        >>> d = _kv_split("I=3, J=4, K=1")

    """
    result: dict[str, str] = {}
    text = text.replace("\n", " ").strip().rstrip(",")
    i, n = 0, len(text)

    while i < n:
        # Skip separators
        while i < n and text[i] in " \t,":
            i += 1
        if i >= n:
            break

        # Key
        key_start = i
        while i < n and text[i] not in "= \t,":
            i += 1
        key = text[key_start:i].strip().upper()
        if not key:
            i += 1
            continue

        # Determine the separator between key and value. The modern form is ``KEY =
        # VALUE``; some legacy files (and ``preplot``) also accept a bare whitespace
        # separator, e.g. ``STRANDID 2``.
        while i < n and text[i] in " \t":
            i += 1
        if i < n and text[i] == "=":
            # Standard ``KEY=VALUE``.
            i += 1
            while i < n and text[i] in " \t":
                i += 1
        elif i >= n or not _next_token_is_value(text, i):
            # Bare flag keyword with no value (e.g. at end of header or directly before
            # another ``KEY=`` pair).
            result[key] = ""
            continue
        # else: legacy ``KEY VALUE`` form — ``i`` already points at the value.

        # Value
        if i >= n:
            result[key] = ""
            continue

        if text[i] == '"':
            j = i + 1
            while j < n and not (text[j] == '"' and text[j - 1] != "\\"):
                j += 1
            value = text[i + 1 : j].replace('\\"', '"')
            i = j + 1
        elif text[i] == "(":
            depth, j = 0, i
            while j < n:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            value = text[i:j]
            i = j
        elif text[i] == "[":
            j = text.find("]", i)
            value = text[i : j + 1] if j >= 0 else text[i:]
            i = j + 1 if j >= 0 else n
        else:
            j = i
            while j < n and text[j] not in " \t,":
                j += 1
            value = text[i:j]
            i = j

        result[key] = value.strip()

    return result


def _parse_legacy_format(text: str) -> tuple[bool, DataPacking]:
    """Parse a legacy ``F=`` value into ``(is_fe, DataPacking)``.

    The ``F`` keyword predates the modern ``ZONETYPE`` / ``DATAPACKING`` pair and
    encodes both pieces of information at once (``POINT``/``BLOCK`` for ordered zones,
    ``FEPOINT``/``FEBLOCK`` for finite-element zones).

    Example:
        >>> is_fe, packing = _parse_legacy_format("FEPOINT")

    Args:
        text: Raw ``F`` value string, e.g. ``"FEPOINT"`` (case-insensitive, any trailing
            comma is ignored).

    Returns:
        Tuple of ``(is_fe, packing)`` where *is_fe* is ``True`` for ``FEPOINT`` /
        ``FEBLOCK`` and *packing* is the corresponding :class:`DataPacking`.

    Raises:
        ValueError: If *text* is not one of the four recognised legacy formats.
    """
    raw = text.rstrip(",").strip().upper()
    try:
        return _STR_TO_LEGACY_FORMAT[raw]
    except KeyError:
        raise ValueError(
            f"Unrecognised legacy zone format F={text!r}; expected one of "
            "POINT, BLOCK, FEPOINT, FEBLOCK."
        ) from None


def _parse_legacy_element_type(text: str) -> ZoneType:
    """Parse a legacy ``ET=`` (element type) value into a :class:`ZoneType`.

    Recognises the canonical Tecplot element names (``TRIANGLE``, ``QUADRILATERAL``,
    ``TETRAHEDRON``, ``BRICK``, ``LINESEG``). A prefix-based fallback provides a little
    tolerance for the minor spelling variants some third-party exporters emit, keeping
    the reader as permissive as ``preplot``.

    Example:
        >>> zt = _parse_legacy_element_type("QUADRILATERAL")

    Args:
        text: Raw ``ET`` value string (case-insensitive, any trailing comma is
            ignored).

    Returns:
        The matching finite-element :class:`ZoneType`.

    Raises:
        ValueError: If *text* does not resemble any known element type.
    """
    raw = text.rstrip(",").strip().upper()
    if raw in _STR_TO_ELEMENT_TYPE:
        return _STR_TO_ELEMENT_TYPE[raw]
    # Prefix-based fallback for minor spelling variants.
    if raw.startswith("TRI"):
        return ZoneType.FETRIANGLE
    if raw.startswith("QUAD"):
        return ZoneType.FEQUADRILATERAL
    if raw.startswith("TET"):
        return ZoneType.FETETRAHEDRON
    if raw.startswith("BRICK"):
        return ZoneType.FEBRICK
    if raw.startswith("LINE"):
        return ZoneType.FELINESEG
    raise ValueError(f"Unrecognised legacy element type ET={text!r} in ZONE header.")


def _parse_index_list(text: str) -> list[int]:
    """Parse a bracketed index list (with optional range notation) into 0-based ints.

    Handles both comma-separated individuals and inclusive ranges::

        "[1,3,5]"    → [0, 2, 4]
        "[1-3,5]"    → [0, 1, 2, 4]
        "[2-4]"      → [1, 2, 3]

    Args:
        text: Raw value string including the surrounding brackets, e.g.
              ``"[1-3,5]"`` or ``"[2]"``.

    Returns:
        List of 0-based variable indices in the order they were listed.

    """
    text = text.strip().lstrip("[").rstrip("]")
    result: list[int] = []
    for tok in re.split(r"[,\s]+", text):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            # Range notation: "N-M" (both 1-based, inclusive).
            parts = tok.split("-", 1)
            with contextlib.suppress(ValueError):
                start_1 = int(parts[0])
                end_1 = int(parts[1])
                result.extend(range(start_1 - 1, end_1))  # → 0-based
        else:
            with contextlib.suppress(ValueError):
                result.append(int(tok) - 1)  # 1-based → 0-based
    return result


def _parse_varlocation(text: str) -> dict[int, ValueLocation]:
    """Parse ``VARLOCATION=([i-j,...]=CELLCENTERED)`` → ``{0-based: loc}``.

    Handles both individual indices and inclusive range notation::

        "([3,4]=CELLCENTERED)"   → {2: CELL_CENTERED, 3: CELL_CENTERED}
        "([3-5]=CELLCENTERED)"   → {2: CELL_CENTERED, 3: CELL_CENTERED,
                                    4: CELL_CENTERED}

    Args:
        text: Raw VARLOCATION value string, e.g.
              ``"([3-5]=CELLCENTERED)"``.

    Returns:
        Dict mapping 0-based variable index to :class:`ValueLocation`.

    """
    result: dict[int, ValueLocation] = {}
    for m in re.finditer(r"\[([^\]]+)\]\s*=\s*(\w+)", text):
        indices_str = m.group(1)
        loc_str = m.group(2).upper()
        loc = (
            ValueLocation.CELL_CENTERED
            if loc_str == "CELLCENTERED"
            else ValueLocation.NODAL
        )
        for idx_0based in _parse_index_list(f"[{indices_str}]"):
            result[idx_0based] = loc
    return result


def _parse_varsharelist(text: str) -> dict[int, int]:
    """Parse ``VARSHARELIST=([i-j]=z,...)`` → ``{0-based var: 1-based zone}``.

    Handles both individual indices and inclusive range notation::

        "([1]=2,[3]=2)"   → {0: 2, 2: 2}
        "([1-2]=1)"       → {0: 1, 1: 1}
        "([1-3,5]=2)"     → {0: 2, 1: 2, 2: 2, 4: 2}

    Note that the bracketed group may contain either a single integer or a
    hyphen-separated range. Mixed forms such as ``[1-3,5]`` are also produced by some
    writers.

    Args:
        text: Raw VARSHARELIST value string including its outer parentheses,
              e.g. ``"([1-2]=1,[4]=2)"``.

    Returns:
        Dict mapping 0-based variable index to 1-based source zone number.
    """
    result: dict[int, int] = {}
    # Each match captures one bracketed group and the zone it maps to.
    # Group 1: everything inside [...] — may be "N", "N-M", or "N-M,P,..."
    # Group 2: the zone number after "=".
    for m in re.finditer(r"\[([^\]]+)\]\s*=\s*(\d+)", text):
        zone_1based = int(m.group(2))
        indices_str = m.group(1)
        # Reuse _parse_index_list to handle ranges and comma lists uniformly.
        for idx_0based in _parse_index_list(f"[{indices_str}]"):
            result[idx_0based] = zone_1based
    return result


def _parse_dt(text: str) -> list[DataType]:
    """Parse ``DT=(SINGLE SINGLE ...)`` -> one :class:`DataType` per variable.

    Note:
        Tokens may be separated by whitespace, commas, or both.

    Args:
        text: Raw DT value string including its outer parentheses, e.g. ``"(SINGLE,
              SINGLE, DOUBLE)"``.

    Returns:
        One :class:`DataType` per variable, in dataset variable order.

    Raises:
        ValueError: If a token isn't a recognized Tecplot data type keyword.

    Example:
        >>> _parse_dt("(SINGLE SINGLE DOUBLE)")
        [DataType.FLOAT, DataType.FLOAT, DataType.DOUBLE]
    """
    inner = text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    tokens = [tok for tok in re.split(r"[,\s]+", inner) if tok]
    result: list[DataType] = []
    for tok in tokens:
        dt = _STR_TO_DATATYPE.get(tok.strip().lower())
        if dt is None:
            raise ValueError(
                f"DT={text!r} contains unrecognized data type {tok!r}; expected one "
                "of SINGLE, DOUBLE, LONGINT, SHORTINT, BYTE."
            )
        result.append(dt)
    return result


def _parse_auxdata_line(line: str) -> tuple[str, str]:
    """Parse ``DATASETAUXDATA name="value"`` → ``(name, value)``.

    Example:
        >>> name, val = _parse_auxdata_line('DATASETAUXDATA Solver="MyCFD"')
    """
    m = re.match(r"(?i)DATASETAUXDATA\s+", line)
    rest = line[m.end() :] if m else line
    eq = rest.find("=")
    if eq < 0:
        return "", ""
    return rest[:eq].strip(), _unquote(rest[eq + 1 :].strip())


def _apply_varauxdata(line: str, var_auxdata_list: list[dict[str, str]]) -> None:
    """Parse ``VARAUXDATA 1-based-idx name="value"`` and store in list.

    *var_auxdata_list* holds plain, mutable dicts, one per variable, 0-based,
    accumulated during parsing. Wrapped into an immutable TecplotDatAuxDataReader
    lazily, only once parsing is complete.

    Example:
        >>> _apply_varauxdata(line, var_auxdata_list)
    """
    m = re.match(r"(?i)VARAUXDATA\s+(\d+)\s+", line)
    if not m:
        return
    var_idx = int(m.group(1)) - 1  # 1-based in text, converted to 0-based
    rest = line[m.end() :]
    eq = rest.find("=")
    if eq < 0:
        return
    name = rest[:eq].strip()
    value = _unquote(rest[eq + 1 :].strip())
    if 0 <= var_idx < len(var_auxdata_list):
        var_auxdata_list[var_idx][name] = value


class _LineBuffer:
    """Peekable iterator over stripped, comment-free text lines.

    Also exposes a small *raw* interface (:meth:`take_raw`, :meth:`position`,
    :meth:`seek`) used only by the vectorized numeric-block readers
    (:meth:`TecplotDatReader._read_float_block_fast`/
    :meth:`TecplotDatReader._read_int_block_fast`). Those readers bypass per-line
    comment-stripping for speed and instead detect the rare line that *isn't* plain
    numeric data (a comment, a header keyword) by letting NumPy's parser fail on it,
    then falling back to this class's ordinary stripped-line interface. Keeping both
    interfaces on one object means the fast path can hand back an exact resume position
    for that fallback via :meth:`position`/:meth:`seek`.

    Example:
        >>> buf = _LineBuffer(lines)
    """

    __slots__ = ("_lines", "_peeked", "_peeked_pos", "_pos")

    def __init__(self, lines: list[str]) -> None:
        self._lines: list[str] = lines
        self._pos: int = 0
        # One-line lookahead cache so a peek_stripped() immediately followed by
        # next_stripped() strips comments/whitespace only once instead of twice.
        self._peeked: str | None = None
        self._peeked_pos: int = -1

    def has_more(self) -> bool:
        """Return ``True`` if there are unconsumed lines."""
        return self._peeked is not None or self._pos < len(self._lines)

    def peek_stripped(self) -> str:
        """Return the next non-blank stripped line without consuming it."""
        if self._peeked is not None:
            return self._peeked
        pos = self._pos
        n = len(self._lines)
        while pos < n:
            stripped = _strip_comment(self._lines[pos]).strip()
            pos += 1
            if stripped:
                self._peeked = stripped
                self._peeked_pos = pos
                return stripped
        self._peeked = ""
        self._peeked_pos = pos
        return ""

    def next_stripped(self) -> str:
        """Consume and return the next non-blank stripped line."""
        if self._peeked is None:
            self.peek_stripped()
        value = self._peeked or ""
        self._pos = self._peeked_pos
        self._peeked = None
        self._peeked_pos = -1
        return value

    def skip_blank(self) -> None:
        """Advance past blank and comment-only lines.

        The vectorized integer readers parse raw lines, so a ``# face nodes`` style
        section comment must be skipped before them or they fall back to the slow path.
        """
        self._peeked = None
        self._peeked_pos = -1
        pos, n = self._pos, len(self._lines)
        while pos < n and not _strip_comment(self._lines[pos]).strip():
            pos += 1
        self._pos = pos

    def take_raw(self, count: int) -> list[str]:
        """Consume and return up to *count* **unprocessed** source lines.

        Unlike :meth:`next_stripped`, this does not strip comments, skip blank lines, or
        strip whitespace -- it is a raw slice of the underlying line list, advancing the
        position by exactly how many lines were returned (fewer than *count* at end of
        file). Any pending :meth:`peek_stripped` lookahead is discarded first, since
        this call always resumes from the true unconsumed position.

        Intended only for the numeric fast-path readers, which feed the result straight
        to a whitespace-tolerant NumPy parser and don't need per-line preprocessing.

        Example:
            >>> lines = buf.take_raw(64)
        """
        self._peeked = None
        self._peeked_pos = -1
        end = min(self._pos + count, len(self._lines))
        out = self._lines[self._pos : end]
        self._pos = end
        return out

    def position(self) -> int:
        """Return an opaque marker for the current position.

        Example:
            >>> marker = buf.position()
        """
        return self._pos

    def seek(self, marker: int) -> None:
        """Restore the position to a marker previously returned by :meth:`position`.

        Example:
            >>> buf.seek(marker)
        """
        self._pos = marker
        self._peeked = None
        self._peeked_pos = -1


# ======================================================================================
# TecplotDatAuxDataReader
# ======================================================================================


class TecplotDatAuxDataReader(TecplotAuxDataReader):
    """Auxiliary data for DAT files.

    Dataset-, zone-, and variable-level aux data are all accumulated into plain dicts
    during parsing (see :class:`TecplotDatReader`), this just wraps one of them.
    """

    __slots__ = ("_raw",)
    _raw: dict[str, str]

    def __init__(self, data: dict[str, str]) -> None:
        super().__init__()
        object.__setattr__(self, "_raw", data)

    def _load_data(self) -> dict[str, str]:
        return self._raw


# ======================================================================================
# TecplotDatVariableReader
# ======================================================================================


class TecplotDatVariableReader(TecplotVariableReader):
    """Variable reader for ASCII DAT files.

    The entire file is parsed into memory up front, so every property here is a plain
    stored value, no disk I/O happens after construction. ``is_enabled`` is not
    overridden here, DAT has no dataset-level enabled flag, so the base class default
    (``not is_passive()``) applies.
    """

    __slots__ = (
        "zone_index",
        "var_index",
        "_name",
        "_array",
        "_value_location",
        "_passive",
        "_shared_zone",
    )
    zone_index: int
    var_index: int
    _name: str
    _array: npt.NDArray[Any] | None
    _value_location: ValueLocation
    _passive: bool
    _shared_zone: int | None

    def __init__(
        self,
        zone_index: int,
        var_index: int,
        name: str,
        data: npt.NDArray[Any] | None,
        value_location: ValueLocation = ValueLocation.NODAL,
        is_passive: bool = False,
        shared_zone: int | None = None,
    ) -> None:
        object.__setattr__(self, "zone_index", zone_index)
        object.__setattr__(self, "var_index", var_index)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_array", data)
        object.__setattr__(self, "_value_location", value_location)
        object.__setattr__(self, "_passive", is_passive)
        object.__setattr__(self, "_shared_zone", shared_zone)

    @property
    def name(self) -> str:
        """Variable name string."""
        return self._name

    @property
    def data_type(self) -> DataType:
        """Data type inferred from the stored NumPy dtype.

        A shared variable reports the source zone's actual dtype, since its data array
        is resolved from that zone. Only a passive variable (which has no data array
        anywhere in the file) returns DataType.FLOAT as a placeholder.
        """
        if self._array is None:
            return DataType.FLOAT
        dt = self._array.dtype
        if dt == np.float64:
            return DataType.DOUBLE
        if dt == np.float32:
            return DataType.FLOAT
        if dt == np.int32:
            return DataType.INT32
        if dt == np.int16:
            return DataType.INT16
        if dt == np.uint8:
            return DataType.BYTE
        return DataType.DOUBLE if dt.itemsize >= 8 else DataType.FLOAT

    @property
    def value_location(self) -> ValueLocation:
        """Value location (NODAL or CELL_CENTERED)."""
        return self._value_location

    def is_passive(self) -> bool:
        """True if this variable is passive (no data in this zone)."""
        return self._passive

    @property
    def shared_zone(self) -> int | None:
        """0-based source zone index if shared, else None."""
        return self._shared_zone

    @property
    def num_values(self) -> int:
        """Number of values stored for this variable."""
        return 0 if self._array is None else self._array.size

    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> npt.NDArray[Any] | None:
        """Return a slice of the data array using a 1-based range.

        Args:
            value_range: 1-based ``(start, end)``. ``(None, None)`` reads all
                values.

        Returns:
            The array (or a slice of it), or None for a passive variable. A shared
            variable resolves to the source zone's array, per :attr:`values`.

        Raises:
            ValueError: If only one of start/end is given.
        """
        if self._array is None:
            return None
        start, end = value_range
        if start is None and end is None:
            return self._array
        if start is None or end is None:
            raise ValueError("Both start and end indices must be specified.")
        return self._array[start - 1 : end - 1]


# ======================================================================================
# TecplotDatOrderedZoneReader / TecplotDatFEZoneReader
# ======================================================================================


class TecplotDatOrderedZoneReader(TecplotOrderedZoneReader):
    """Ordered (IJK) zone reader for ASCII DAT files.

    The entire zone (variable arrays, aux data) is already fully parsed and held in
    memory by the time this is constructed, so the base class's lazy-loading hooks below
    do no real work, they just wrap already-computed values on first access, for
    interface consistency with SZL and PLT.  ``is_enabled`` is not overridden here, DAT
    has no concept of a disabled zone, so the base class default (always True) applies.
    """

    __slots__ = ("_variables", "_aux_raw")
    _variables: list[TecplotVariableReader]
    _aux_raw: dict[str, str]

    def __init__(
        self,
        zone_index: int,
        title: str,
        I: int,  # noqa: E741 N803
        J: int,  # noqa: N803
        K: int,  # noqa: N803
        solution_time: float,
        strand_id: int,
        variables: list[TecplotVariableReader],
        auxdata: dict[str, str],
        datapacking: DataPacking = DataPacking.BLOCK,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=title,
            solution_time=solution_time,
            strand_id=strand_id,
            datapacking=datapacking,
            i=I,
            j=J,
            k=K,
        )
        object.__setattr__(self, "_variables", variables)
        object.__setattr__(self, "_aux_raw", auxdata)

    def _load_variables(self) -> list[TecplotVariableReader]:
        return self._variables

    def _load_auxdata(self) -> TecplotAuxDataReader:
        return TecplotDatAuxDataReader(self._aux_raw)


class TecplotDatFEZoneReader(TecplotFEZoneReader):
    """Finite-element zone reader for ASCII DAT files.

    The entire zone (variable arrays, node map, aux data) is already fully parsed and
    held in memory by the time this is constructed, so the base class's lazy-loading
    hooks below do no real work, they just wrap already-computed values on first access,
    for interface consistency with SZL and PLT. ``is_enabled`` is not overridden here,
    DAT has no concept of a disabled zone, so the base class default (always True)
    applies.

    FEPOLYGON/FEPOLYHEDRON zones are read as :class:`TecplotDatPolyZoneReader`.
    """

    __slots__ = (
        "_variables",
        "_node_map",
        "_aux_raw",
        "_face_neighbor_mode",
        "_num_face_connections",
        "_face_connections",
        "_face_neighbors_complete",
    )
    _variables: list[TecplotVariableReader]
    _node_map: npt.NDArray[Any] | None
    _aux_raw: dict[str, str]
    _face_neighbor_mode: FaceNeighborMode | None
    _num_face_connections: int | None
    _face_connections: npt.NDArray[np.int64] | None
    _face_neighbors_complete: bool | None

    def __init__(
        self,
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        num_nodes: int,
        num_elements: int,
        solution_time: float,
        strand_id: int,
        variables: list[TecplotVariableReader],
        auxdata: dict[str, str],
        node_map: npt.NDArray[Any] | None = None,
        datapacking: DataPacking = DataPacking.BLOCK,
        shared_connectivity: int | None = None,
        face_neighbor_mode: FaceNeighborMode | None = None,
        num_face_connections: int | None = None,
        face_connections: npt.NDArray[np.int64] | None = None,
        face_neighbors_complete: bool | None = None,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=title,
            zone_type=zone_type,
            solution_time=solution_time,
            strand_id=strand_id,
            datapacking=datapacking,
            num_nodes=num_nodes,
            num_elements=num_elements,
            shared_connectivity=shared_connectivity,
        )
        object.__setattr__(self, "_variables", variables)
        object.__setattr__(self, "_node_map", node_map)
        object.__setattr__(self, "_aux_raw", auxdata)
        object.__setattr__(self, "_face_neighbor_mode", face_neighbor_mode)
        object.__setattr__(self, "_num_face_connections", num_face_connections)
        object.__setattr__(self, "_face_connections", face_connections)
        object.__setattr__(self, "_face_neighbors_complete", face_neighbors_complete)

    def _load_variables(self) -> list[TecplotVariableReader]:
        return self._variables

    def _load_node_map(self) -> npt.NDArray[Any] | None:
        return self._node_map

    def _load_face_neighbor_meta(
        self,
    ) -> tuple[FaceNeighborMode, int, bool | None] | None:
        if self._face_neighbor_mode is None or self._num_face_connections is None:
            return None
        return (
            self._face_neighbor_mode,
            self._num_face_connections,
            self._face_neighbors_complete,
        )

    def _load_face_connections(self) -> npt.NDArray[np.int64]:
        assert self._face_connections is not None  # narrowed: meta already confirmed
        return self._face_connections

    def _load_auxdata(self) -> TecplotAuxDataReader:
        return TecplotDatAuxDataReader(self._aux_raw)


class TecplotDatPolyZoneReader(TecplotPolyZoneReader):
    """Poly (FEPOLYGON / FEPOLYHEDRON) zone reader for ASCII DAT files.

    Like :class:`TecplotDatFEZoneReader`, the whole zone is already parsed and held in
    memory at construction, so the lazy hooks just hand back stored arrays. The face
    map is already converted to the public 0-based convention.
    """

    __slots__ = ("_variables", "_aux_raw", "_face_map")
    _variables: list[TecplotVariableReader]
    _aux_raw: dict[str, str]
    _face_map: dict[str, npt.NDArray[np.int64]]

    def __init__(
        self,
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        num_nodes: int,
        num_elements: int,
        num_faces: int,
        num_boundary_faces: int,
        num_boundary_connections: int,
        solution_time: float,
        strand_id: int,
        variables: list[TecplotVariableReader],
        auxdata: dict[str, str],
        face_map: dict[str, npt.NDArray[np.int64]],
        datapacking: DataPacking = DataPacking.BLOCK,
        shared_connectivity: int | None = None,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=title,
            zone_type=zone_type,
            solution_time=solution_time,
            strand_id=strand_id,
            datapacking=datapacking,
            num_nodes=num_nodes,
            num_elements=num_elements,
            num_faces=num_faces,
            num_boundary_faces=num_boundary_faces,
            num_boundary_connections=num_boundary_connections,
            shared_connectivity=shared_connectivity,
        )
        object.__setattr__(self, "_variables", variables)
        object.__setattr__(self, "_aux_raw", auxdata)
        object.__setattr__(self, "_face_map", face_map)

    def _load_variables(self) -> list[TecplotVariableReader]:
        return self._variables

    def _load_array(self, key: PolyArrayKey) -> npt.NDArray[np.int64]:
        return self._face_map[key]

    def _load_auxdata(self) -> TecplotAuxDataReader:
        return TecplotDatAuxDataReader(self._aux_raw)


class _DatZoneInfo(NamedTuple):
    """Per-zone metadata from a cheap header-only scan."""

    title: str
    zone_type: ZoneType
    num_nodes: int | None
    num_elements: int | None


class _DatMetadata(NamedTuple):
    """Result of :func:`_peek_dat_metadata`."""

    title: str
    file_type: FileType
    variable_names: list[str]
    zones: list[_DatZoneInfo]


_ZONE_LINE_RE = re.compile(r"^[ \t]*ZONE\b", re.IGNORECASE | re.MULTILINE)
_ZONE_DATA_START_RE = re.compile(r"^[ \t]*[0-9+\-.]", re.MULTILINE)


def _peek_dat_metadata(path: str | os.PathLike) -> _DatMetadata:
    """Scan a DAT file's title, variable names, and per-zone metadata only.

    Unlike opening the file normally, never converts a single data value to a number. A
    single bulk regex pass across the whole file finds every zone header's starting
    line; each zone's metadata is then extracted from just the handful of lines between
    that start and the next one (or EOF), reusing the same header-parsing logic the real
    parser uses, so a zone header can still span multiple lines correctly. The data and
    connectivity blocks in between are never read at all. A line-by-line Python scan
    over the whole file (rather than this bulk approach) was tried first and measured
    slower than a full parse for large files, string-heavy per-line loops in Python
    don't beat NumPy's bulk numeric parsing, even when doing less work; a single C-level
    regex pass over the whole file does.

    A zone's ``num_nodes``/``num_elements`` are ``None`` only for the one case they
    genuinely can't be determined without reading data: an ordered zone whose header
    omits I/J/K entirely, relying on the reader to infer the point count from the data
    itself (see :meth:`TecplotDatReader._infer_ordered_point_count`). Every other case,
    including FE zones, always states its node/element counts directly in the header
    text.

    Args:
        path: Path to a ``.dat``/``.tec`` file.

    Returns:
        Title, file type, variable names, and per-zone metadata (in file order).

    Raises:
        FileNotFoundError: If *path* does not exist.

    Example:
        >>> meta = _peek_dat_metadata("flow.dat")
        >>> [z.title for z in meta.zones]
        ['FluidVolume', 'WingSurface']
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    zone_starts = [m.start() for m in _ZONE_LINE_RE.finditer(text)]

    # File header: everything before the first zone (or the whole file, if it somehow
    # has none). Mirrors TecplotDatReader._parse_file_header, kept separate rather than
    # shared with it since that method also populates dataset aux data, which this
    # function deliberately doesn't scan for. Just a handful of lines regardless of file
    # size, so no bulk-regex treatment needed here.
    header_end = zone_starts[0] if zone_starts else len(text)
    tokens = _LineBuffer(text[:header_end].splitlines(keepends=True))

    title = ""
    file_type = FileType.FULL
    variable_names: list[str] = []
    while tokens.has_more():
        line = tokens.peek_stripped()
        if not line:
            tokens.next_stripped()
            continue
        upper_key = (
            line.split("=")[0].strip().upper() if "=" in line else line.strip().upper()
        )
        tokens.next_stripped()

        if upper_key == "TITLE":
            rhs = line.split("=", 1)[1].strip() if "=" in line else ""
            title = _unquote(rhs)
        elif upper_key == "FILETYPE":
            rhs = line.split("=", 1)[1].strip() if "=" in line else ""
            file_type = _STR_TO_FILETYPE.get(rhs.strip().lower(), FileType.FULL)
        elif upper_key == "VARIABLES":
            rhs = line.split("=", 1)[1].strip() if "=" in line else ""
            names = _extract_quoted_strings(rhs)
            while tokens.has_more():
                nxt = tokens.peek_stripped()
                if not nxt:
                    tokens.next_stripped()
                    continue
                if nxt.lstrip().startswith('"'):
                    names.extend(_extract_quoted_strings(tokens.next_stripped()))
                elif "=" not in nxt.split("#")[0]:
                    tokens.next_stripped()
                else:
                    break
            variable_names = names

    # Zones: each zone's header lives in the small window between its start and the next
    # zone's start (or EOF); _collect_zone_header_lines only ever looks at the first
    # handful of lines in that window before hitting what looks like data, so this stays
    # cheap regardless of how much data actually follows.
    zones: list[_DatZoneInfo] = []
    for i, start in enumerate(zone_starts):
        end = zone_starts[i + 1] if i + 1 < len(zone_starts) else len(text)
        # Bounded search directly on the original string
        data_match = _ZONE_DATA_START_RE.search(text, start, end)
        header_end = data_match.start() if data_match else end
        window_tokens = _LineBuffer(text[start:header_end].splitlines(keepends=True))
        header_lines = _collect_zone_header_lines(window_tokens)
        header_text = " ".join(header_lines)
        m_zone = re.match(r"(?i)^ZONE\s*", header_text)
        if m_zone:
            header_text = header_text[m_zone.end() :]
        kv = _kv_split(header_text)
        zone_title = _unquote(kv.get("T", ""))

        try:
            zone_type = _determine_zone_type(kv)
        except ValueError:
            # A legacy header F=FEPOINT with no ET reports as unknown
            zones.append(_DatZoneInfo(zone_title, ZoneType.ORDERED, None, None))
            continue

        num_nodes: int | None
        num_elements: int | None
        if zone_type == ZoneType.ORDERED:
            if "I" not in kv and "J" not in kv and "K" not in kv:
                # Point count only knowable by reading data -> report unknown
                num_nodes = None
                num_elements = None
            else:
                zi = int(kv.get("I", "1") or "1")
                zj = int(kv.get("J", "1") or "1")
                zk = int(kv.get("K", "1") or "1")
                num_nodes = zi * zj * zk
                num_elements = max(zi - 1, 1) * max(zj - 1, 1) * max(zk - 1, 1)
        else:
            # FE and poly zones always state counts directly.
            num_nodes = int(kv.get("NODES", kv.get("N", "0")) or "0")
            num_elements = int(kv.get("ELEMENTS", kv.get("E", "0")) or "0")

        zones.append(_DatZoneInfo(zone_title, zone_type, num_nodes, num_elements))

    return _DatMetadata(title, file_type, variable_names, zones)


def _face_map_of(zone: TecplotPolyZoneReader) -> dict[str, npt.NDArray[np.int64]]:
    """Return the face-map arrays of an already parsed poly zone."""
    face_map = {
        "face_nodes": zone.face_nodes,
        "face_left_elements": zone.face_left_elements,
        "face_right_elements": zone.face_right_elements,
    }
    if zone.zone_type is ZoneType.FEPOLYHEDRON:
        face_map["face_node_counts"] = zone.face_node_counts
    if zone.num_boundary_faces:
        face_map["boundary_connection_counts"] = zone.boundary_connection_counts
        face_map["boundary_connection_elements"] = zone.boundary_connection_elements
        face_map["boundary_connection_zones"] = zone.boundary_connection_zones
    return face_map


class TecplotDatReader(TecplotReader):
    """Reader for Tecplot ASCII DAT files.

    The entire file is parsed into memory at construction. Holds no open file handle
    between accesses, :meth:`close` is a no-op (inherited from the base class default),
    same as PLT.

    Args:
        path: Path to the ``.dat`` file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: On unsupported format features.
    """

    __slots__ = (
        "_path",
        "_title",
        "_file_type",
        "_variable_names",
        "_zones",
        "_zone_list",
        "_dataset_auxdata_raw",
        "_dataset_auxdata",
        "_var_auxdata_raw",
        "_deferred_var_aux_lines",
    )

    def __init__(self, path: str) -> None:
        self._path: str = str(path)
        self._title: str = ""
        self._file_type: FileType = FileType.FULL
        self._variable_names: list[str] = []
        self._zones: list[TecplotZoneReader] = []
        self._zone_list: ZoneList[TecplotZoneReader] | None = None
        # Accumulated as plain dicts during parsing (aux lines can appear anywhere in
        # the file, interleaved with zones), wrapped into an immutable
        # TecplotDatAuxDataReader lazily, only once fully parsed.
        self._dataset_auxdata_raw: dict[str, str] = {}
        self._dataset_auxdata: TecplotAuxDataReader | None = None
        # 0-based, one dict per variable; populated once num_vars is known, see
        # _parse().
        self._var_auxdata_raw: list[dict[str, str]] = []
        # Raw VARAUXDATA lines seen before the first zone, buffered by
        # _parse_file_header() and applied once _var_auxdata_raw is allocated with the
        # correct length (num_vars isn't known until the header finishes parsing).
        self._deferred_var_aux_lines: list[str] = []
        self._parse()

    @property
    def path(self) -> str:
        """Source file path."""
        return self._path

    @property
    def file_type(self) -> FileType:
        """Return :class:`FileType` enum for this file."""
        return self._file_type

    @property
    def title(self) -> str:
        """Return the dataset title string."""
        return self._title

    @property
    def num_vars(self) -> int:
        """Return the number of variables in the dataset."""
        return len(self._variable_names)

    @property
    def variables(self) -> list[str]:
        """Return the ordered list of variable name strings."""
        return list(self._variable_names)

    @property
    def num_zones(self) -> int:
        """Return the number of zones in the file."""
        return len(self._zones)

    @property
    def zones(self) -> ZoneList[TecplotZoneReader]:
        """Zones in this file, by index or slice."""
        if self._zone_list is None:
            self._zone_list = ZoneList(self._zones)
        return self._zone_list

    @property
    def auxdata(self) -> TecplotAuxDataReader:
        """Return the dataset-level auxiliary data."""
        if self._dataset_auxdata is None:
            self._dataset_auxdata = TecplotDatAuxDataReader(self._dataset_auxdata_raw)
        return self._dataset_auxdata

    def _var_auxdata_at(self, var_index: int) -> TecplotAuxDataReader:
        return TecplotDatAuxDataReader(self._var_auxdata_raw[var_index])

    # -- parser ------------------------------------------------------------------------

    def _parse(self) -> None:
        """Read and parse the entire DAT file.

        Example:
            >>> self._parse()
        """
        with open(self._path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        tokens = _LineBuffer(lines)
        self._parse_file_header(tokens)

        # Build per-variable aux data slots now that num_vars is known.
        self._var_auxdata_raw = [{} for _ in range(self.num_vars)]

        # Now that _var_auxdata exists, apply any VARAUXDATA lines that appeared before
        # the first zone
        for raw in self._deferred_var_aux_lines:
            _apply_varauxdata(raw, self._var_auxdata_raw)
        self._deferred_var_aux_lines.clear()

        while tokens.has_more():
            line = tokens.peek_stripped()
            if not line:
                tokens.next_stripped()
                continue
            upper = line.upper()
            if upper.lstrip().split("=")[0].split()[0] == "ZONE":
                self._parse_zone(tokens)
            elif upper.startswith("DATASETAUXDATA"):
                raw = tokens.next_stripped()
                name, value = _parse_auxdata_line(raw)
                if name:
                    self._dataset_auxdata_raw[name] = value
            elif upper.startswith("VARAUXDATA"):
                raw = tokens.next_stripped()
                _apply_varauxdata(raw, self._var_auxdata_raw)
            else:
                tokens.next_stripped()  # skip unrecognised lines

    def _parse_file_header(self, tokens: _LineBuffer) -> None:
        """Parse TITLE, FILETYPE, VARIABLES, and dataset-level aux data.

        Stops (without consuming) when a ZONE keyword is seen.

        Example:
            >>> self._parse_file_header(tokens)
        """
        while tokens.has_more():
            line = tokens.peek_stripped()
            upper_key = (
                line.split("=")[0].strip().upper()
                if "=" in line
                else line.strip().upper()
            )

            if upper_key.startswith("ZONE"):
                break

            tokens.next_stripped()  # consume

            if upper_key == "TITLE":
                rhs = line.split("=", 1)[1].strip() if "=" in line else ""
                self._title = _unquote(rhs)

            elif upper_key == "FILETYPE":
                rhs = line.split("=", 1)[1].strip() if "=" in line else ""
                self._file_type = _STR_TO_FILETYPE.get(
                    rhs.strip().lower(), FileType.FULL
                )

            elif upper_key == "VARIABLES":
                # Names may be on the same line or on subsequent quoted lines.
                rhs = line.split("=", 1)[1].strip() if "=" in line else ""
                names = _extract_quoted_strings(rhs)

                while tokens.has_more():
                    nxt = tokens.peek_stripped()
                    if not nxt:
                        tokens.next_stripped()
                        continue
                    nxt_upper = nxt.lstrip().upper()
                    # A continuation line is a bare quoted string with no '=' keyword.
                    if nxt.lstrip().startswith('"'):
                        more = _extract_quoted_strings(tokens.next_stripped())
                        names.extend(more)
                    elif (
                        "=" not in nxt.split("#")[0]
                        and not nxt_upper.startswith("ZONE")
                        and not nxt_upper.startswith("DATASETAUXDATA")
                        and not nxt_upper.startswith("VARAUXDATA")
                    ):
                        # Non-quoted, non-keyword line — could be a bare name, skip.
                        tokens.next_stripped()
                    else:
                        break

                self._variable_names = names

            elif upper_key.startswith("DATASETAUXDATA"):
                name, value = _parse_auxdata_line(line)
                if name:
                    self._dataset_auxdata_raw[name] = value

            elif upper_key.startswith("VARAUXDATA"):
                # Can't process this without num_vars, and therefore _var_auxdata, isn't
                # known until this header finishes parsing. Buffer the raw line;
                # _parse() applies these once _var_auxdata is allocated.
                self._deferred_var_aux_lines.append(line)

    @staticmethod
    def _infer_ordered_point_count(tokens: _LineBuffer, active_count: int) -> int:
        """Infer an ordered zone's point count when I/J/K are all omitted.

        Some legacy exporters write a bare ``ZONE F=POINT`` header with no dimensions at
        all, relying on the reader to count data rows itself.  Peeks ahead (never
        consuming) counting numeric tokens up to the next zone/keyword boundary or end
        of file, using the same boundary rule :meth:`_parse_zone` already uses to know
        where a header ends and data begins, then restores the original position so the
        real read (later, driven by the now-known count) starts from the same place.

        Args:
            tokens: Positioned at the first data line.
            active_count: Number of active (non-passive, non-shared) variables in
                dataset order; each data row has exactly this many values.

        Returns:
            The inferred point count.

        Raises:
            ValueError: If the total token count isn't evenly divisible by
                *active_count*, a malformed or truncated data block.

        Example:
            >>> n = TecplotDatReader._infer_ordered_point_count(tokens, 3)
        """
        marker = tokens.position()
        total = 0
        while tokens.has_more():
            line = tokens.peek_stripped()
            if not line:
                tokens.next_stripped()
                continue
            upper = line.lstrip().upper()
            if upper.split("=")[0].split()[0] == "ZONE":
                break
            if upper.startswith(("DATASETAUXDATA", "VARAUXDATA")):
                break
            total += len(tokens.next_stripped().split())
        tokens.seek(marker)

        if active_count <= 0 or total % active_count != 0:
            raise ValueError(
                "Could not infer point count for an ordered zone with no "
                f"I/J/K: found {total} values, not evenly divisible by "
                f"{active_count} active variables."
            )
        return total // active_count

    def _parse_zone(self, tokens: _LineBuffer) -> None:
        """Parse one ZONE block (header + data blocks + connectivity).

        Example:
            >>> self._parse_zone(tokens)
        """
        # -- Collect header lines ------------------------------------------------------

        header_lines = _collect_zone_header_lines(tokens)

        header_text = " ".join(header_lines)

        # Strip the leading ZONE keyword from the header text.
        m_zone = re.match(r"(?i)^ZONE\s*", header_text)
        if m_zone:
            header_text = header_text[m_zone.end() :]

        # -- Parse header key=value pairs ----------------------------------------------

        kv = _kv_split(header_text)

        zone_title = _unquote(kv.get("T", ""))
        strand_id = int(kv.get("STRANDID", "0") or "0")
        solution_time = float(kv.get("SOLUTIONTIME", "0.0") or "0.0")

        # -- Determine zone type and data packing --------------------------------------
        #
        # See _determine_zone_type for the modern/legacy keyword rules this shares with
        # the metadata-only scanner (peek_dat_metadata).
        legacy_packing: DataPacking | None = None
        if "F" in kv:
            _, legacy_packing = _parse_legacy_format(kv["F"])

        zone_type = _determine_zone_type(kv)

        # Passive/shared variables are parsed here (ahead of var_locs/var_types below,
        # which don't need this ordering) because the point-count inference just below
        # needs active_count first.
        passive_set: set[int] = set()
        if "PASSIVEVARLIST" in kv:
            passive_set = set(_parse_index_list(kv["PASSIVEVARLIST"]))

        share_map: dict[int, int] = {}
        if "VARSHARELIST" in kv:
            share_map = _parse_varsharelist(kv["VARSHARELIST"])

        if zone_type == ZoneType.ORDERED:
            if "I" not in kv and "J" not in kv and "K" not in kv:
                # Infer point count
                active_count = self.num_vars - len(passive_set) - len(share_map)
                num_nodes = self._infer_ordered_point_count(tokens, active_count)
                I = num_nodes  # noqa E741
                J = K = 1  # noqa: N806
            else:
                I = int(kv.get("I", "1") or "1")  # noqa: E741 N806
                J = int(kv.get("J", "1") or "1")  # noqa: N806
                K = int(kv.get("K", "1") or "1")  # noqa: N806
                num_nodes = I * J * K
            num_cells = max(I - 1, 1) * max(J - 1, 1) * max(K - 1, 1)
        else:
            # FE zones accept both the modern (NODES/ELEMENTS) and legacy (N/E)
            # spellings for the node and element counts.
            num_nodes = int(kv.get("NODES", kv.get("N", "0")) or "0")
            num_cells = int(kv.get("ELEMENTS", kv.get("E", "0")) or "0")
            I, J, K = num_nodes, num_cells, 0  # noqa E741
        num_faces = int(kv.get("FACES", "0") or "0")
        total_face_nodes = int(kv.get("TOTALNUMFACENODES", "0") or "0")
        num_boundary_faces = int(kv.get("NUMCONNECTEDBOUNDARYFACES", "0") or "0")
        num_boundary_connections = int(
            kv.get("TOTALNUMBOUNDARYCONNECTIONS", "0") or "0"
        )

        # Packing: DATAPACKING wins, then legacy F, else BLOCK (Tecplot default).
        if "DATAPACKING" in kv:
            packing_raw = kv["DATAPACKING"].strip().lower()
            packing = _STR_TO_DATAPACKING.get(packing_raw)
            if packing is None:
                raise ValueError(
                    f"DATAPACKING={packing_raw!r} is not supported; "
                    "only BLOCK and POINT are implemented in the ASCII reader."
                )
        elif legacy_packing is not None:
            packing = legacy_packing
        else:
            packing = DataPacking.BLOCK

        # Tecplot refuses POINT for face-based zones, so match it.
        if zone_type in _FE_POLY and packing != DataPacking.BLOCK:
            raise ValueError(
                "Polygon and polyhedron zones must use BLOCK data packing."
            )

        # Face-neighbor connections (finite-element zones only; ordered zones
        # don't need them, adjacency is already implicit in the I/J/K
        # structure). FACENEIGHBORCONNECTIONS is authoritative for how many
        # values to expect; FEFACENEIGHBORSCOMPLETE says whether that's the
        # entire adjacency picture or a supplement to Tecplot's own
        # auto-detection, read below for face_neighbors_complete.
        declared_face_connections = 0
        face_neighbor_mode: FaceNeighborMode | None = None
        face_neighbors_complete: bool | None = None
        if zone_type != ZoneType.ORDERED and "FACENEIGHBORCONNECTIONS" in kv:
            declared_face_connections = int(kv["FACENEIGHBORCONNECTIONS"] or "0")
            if declared_face_connections > 0:
                mode_raw = kv.get("FACENEIGHBORMODE", "LOCALONETOONE").strip().lower()
                face_neighbor_mode = _STR_TO_FACENEIGHBORMODE.get(
                    mode_raw, FaceNeighborMode.LOCAL_ONE_TO_ONE
                )
                if "FEFACENEIGHBORSCOMPLETE" in kv:
                    complete_raw = kv["FEFACENEIGHBORSCOMPLETE"].strip().upper()
                    face_neighbors_complete = complete_raw == "YES"

        # Variable locations (0-based index → ValueLocation)
        var_locs: dict[int, ValueLocation] = {}
        if "VARLOCATION" in kv:
            var_locs = _parse_varlocation(kv["VARLOCATION"])

        # Per-variable data types. Tecplot default when DT= is omitted is SINGLE for
        # every variable.
        if "DT" in kv:
            var_types = _parse_dt(kv["DT"])
            if len(var_types) != len(self._variable_names):
                raise ValueError(
                    f"DT={kv['DT']!r} declares {len(var_types)} data types, but "
                    f"the dataset has {len(self._variable_names)} variables."
                )
        else:
            var_types = [DataType.FLOAT] * len(self._variable_names)

        # Connectivity sharing
        con_share_zone = int(kv.get("CONNECTIVITYSHAREZONE", "0") or "0")

        # Zone-level aux data (AUXDATA key="value" lines inside the header)
        zone_aux: dict[str, str] = {}
        for m in re.finditer(
            r"(?i)AUXDATA\s+(\S+)\s*=\s*(\"[^\"]*\"|[^\s,\"]+)", header_text
        ):
            zone_aux[m.group(1)] = _unquote(m.group(2))

        # -- Read variable data blocks -------------------------------------------------

        if packing == DataPacking.POINT:
            var_arrays = self._read_point_var_data(
                tokens,
                self.num_vars,
                num_nodes,
                num_cells,
                var_locs,
                passive_set,
                share_map,
                var_types,
            )
        else:
            var_arrays = self._read_block_var_data(
                tokens,
                self.num_vars,
                num_nodes,
                num_cells,
                var_locs,
                passive_set,
                share_map,
                var_types,
            )

        # -- Read connectivity (FE zones only) -----------------------------------------

        node_map: npt.NDArray | None = None
        face_map: dict[str, npt.NDArray[np.int64]] = {}

        if zone_type in _FE_POLY:
            if con_share_zone and self._zones:
                source_zone = self._zones[con_share_zone - 1]
                if isinstance(source_zone, TecplotPolyZoneReader):
                    face_map = _face_map_of(source_zone)
                    num_faces = source_zone.num_faces
                    num_boundary_faces = source_zone.num_boundary_faces
                    num_boundary_connections = source_zone.num_boundary_connections
            else:
                face_map = self._read_poly_face_map(
                    tokens,
                    zone_type,
                    num_faces,
                    total_face_nodes,
                    num_boundary_faces,
                    num_boundary_connections,
                )
        elif zone_type != ZoneType.ORDERED:
            if con_share_zone and self._zones:
                source_zone = self._zones[con_share_zone - 1]
                if isinstance(source_zone, TecplotFEZoneReader):
                    node_map = source_zone.node_map
                # else: malformed file referencing a non-FE zone for shared
                # connectivity; leave node_map as None rather than crashing on an
                # otherwise-recoverable read.
            else:
                nodes_per_cell = _NODES_PER_ELEM[zone_type]
                flat = self._read_int_block(tokens, num_cells * nodes_per_cell)
                # Node numbers are 1-based in ASCII; convert to 0-based indices
                # that point directly into the corresponding coordinate array.
                node_map = flat.reshape(num_cells, nodes_per_cell) - 1

        # -- Read face-neighbor connections (FE zones only, if declared) ---------------
        #
        # Immediately follows connectivity in the file. Values are 1-based in ASCII;
        # cell references, remote-zone references, and the face ordinal are all
        # converted to 0-based below.

        face_connections: npt.NDArray[np.int64] | None = None
        one_to_one = (
            FaceNeighborMode.LOCAL_ONE_TO_ONE,
            FaceNeighborMode.GLOBAL_ONE_TO_ONE,
        )
        if face_neighbor_mode in one_to_one:
            values_per = (
                3 if face_neighbor_mode is FaceNeighborMode.LOCAL_ONE_TO_ONE else 4
            )
            face_connections = self._read_int_block(
                tokens, declared_face_connections * values_per
            )
            face_connections = convert_face_connections_to_zero_based(
                face_connections, face_neighbor_mode
            )
        elif face_neighbor_mode is not None:
            # "Many" modes: ragged, each record's own 4th value (nz) gives the
            # count of additional neighbor references that follow it.
            is_global = face_neighbor_mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
            pieces: list[npt.NDArray] = []
            for _ in range(declared_face_connections):
                header = self._read_int_block(tokens, 4)
                nz = int(header[3])
                rest_count = 2 * nz if is_global else nz
                rest = self._read_int_block(tokens, rest_count)
                pieces.append(header)
                pieces.append(rest)
            face_connections = (
                np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
            )
            face_connections = convert_face_connections_to_zero_based(
                face_connections, face_neighbor_mode
            )

        # -- Build ReadVariable and ReadZone objects -----------------------------------

        # For ordered zones, reshape each variable array from flat 1-D to (I, J, K) for
        # nodal variables or (I-1, J-1, K-1) for cell-centered so that zone dimensions
        # can be inferred from array shape downstream.
        def _shaped(arr, loc):
            if arr is None or zone_type != ZoneType.ORDERED:
                return arr
            if loc == ValueLocation.CELL_CENTERED:
                shape = (max(I - 1, 1), max(J - 1, 1), max(K - 1, 1))
            else:
                shape = (I, J, K)
            if arr.size == shape[0] * shape[1] * shape[2]:
                return arr.reshape(shape, order="F")
            return arr

        # Resolve variable data shared from an earlier zone (VARSHARELIST). Sharing is
        # only valid against a zone that has already been parsed (DAT files reference
        # zones by position, forward-only); an out-of-range reference is left as
        # ``None`` rather than raising, since malformed input shouldn't crash a read
        # that's otherwise recoverable.
        for var_idx, src_zone_1based in share_map.items():
            if 1 <= src_zone_1based <= len(self._zones):
                var_arrays[var_idx] = (
                    self._zones[src_zone_1based - 1].variables[var_idx].values
                )

        # This zone's own 0-based index
        zone_index = len(self._zones)

        read_vars: list[TecplotVariableReader] = [
            TecplotDatVariableReader(
                zone_index=zone_index,
                var_index=idx,
                name=name,
                data=_shaped(
                    var_arrays[idx],
                    var_locs.get(idx, ValueLocation.NODAL),
                ),
                value_location=var_locs.get(idx, ValueLocation.NODAL),
                is_passive=(idx in passive_set),
                shared_zone=(share_map[idx] - 1 if idx in share_map else None),
            )
            for idx, name in enumerate(self._variable_names)
        ]

        if zone_type in _FE_POLY:
            self._zones.append(
                TecplotDatPolyZoneReader(
                    zone_index=zone_index,
                    title=zone_title,
                    zone_type=zone_type,
                    num_nodes=num_nodes,
                    num_elements=num_cells,
                    num_faces=num_faces,
                    num_boundary_faces=num_boundary_faces,
                    num_boundary_connections=num_boundary_connections,
                    solution_time=solution_time,
                    strand_id=strand_id,
                    variables=read_vars,
                    auxdata=zone_aux,
                    face_map=face_map,
                    datapacking=packing,
                    shared_connectivity=(con_share_zone - 1)
                    if con_share_zone
                    else None,
                )
            )
            return

        self._zones.append(
            self._build_zone(
                zone_index=zone_index,
                title=zone_title,
                zone_type=zone_type,
                I=I,
                J=J,
                K=K,
                solution_time=solution_time,
                strand_id=strand_id,
                variables=read_vars,
                auxdata=zone_aux,
                node_map=node_map,
                datapacking=packing,
                shared_connectivity=(con_share_zone - 1) if con_share_zone else None,
                face_neighbor_mode=face_neighbor_mode,
                num_face_connections=declared_face_connections or None,
                face_connections=face_connections,
                face_neighbors_complete=face_neighbors_complete,
            )
        )

    @staticmethod
    def _build_zone(
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        I: int,  # noqa: E741 N803
        J: int,  # noqa: N803
        K: int,  # noqa: N803
        solution_time: float,
        strand_id: int,
        variables: list[TecplotVariableReader],
        auxdata: dict[str, str],
        node_map: npt.NDArray[Any] | None,
        datapacking: DataPacking,
        shared_connectivity: int | None,
        face_neighbor_mode: FaceNeighborMode | None = None,
        num_face_connections: int | None = None,
        face_connections: npt.NDArray[np.int64] | None = None,
        face_neighbors_complete: bool | None = None,
    ) -> TecplotZoneReader:
        """Construct the right concrete zone reader for one parsed zone."""
        if zone_type == ZoneType.ORDERED:
            return TecplotDatOrderedZoneReader(
                zone_index=zone_index,
                title=title,
                I=I,
                J=J,
                K=K,
                solution_time=solution_time,
                strand_id=strand_id,
                variables=variables,
                auxdata=auxdata,
                datapacking=datapacking,
            )
        return TecplotDatFEZoneReader(
            zone_index=zone_index,
            title=title,
            zone_type=zone_type,
            num_nodes=I,
            num_elements=J,
            solution_time=solution_time,
            strand_id=strand_id,
            variables=variables,
            auxdata=auxdata,
            node_map=node_map,
            datapacking=datapacking,
            shared_connectivity=shared_connectivity,
            face_neighbor_mode=face_neighbor_mode,
            num_face_connections=num_face_connections,
            face_connections=face_connections,
            face_neighbors_complete=face_neighbors_complete,
        )

    @staticmethod
    def _read_poly_face_map(
        tokens: _LineBuffer,
        zone_type: ZoneType,
        num_faces: int,
        total_face_nodes: int,
        num_boundary_faces: int,
        num_boundary_connections: int,
    ) -> dict[str, npt.NDArray[np.int64]]:
        """Read the integer face-map sections that follow a poly zone's variables.

        The sections are introduced by ``#`` comments, which are skipped. All indices
        are 1-based in ASCII (0 for no element, ``-k`` for the k-th boundary face), so
        every array is shifted down by one to the public convention.

        Raises:
            ValueError: If a section is shorter than the header declares.
        """

        def section(name: str, count: int, shift: int = 0) -> npt.NDArray[np.int64]:
            tokens.skip_blank()
            arr = TecplotDatReader._read_int_block(tokens, count)
            if arr.size != count:
                raise ValueError(
                    f"Poly zone section '{name}' has {arr.size} values, expected "
                    f"{count}."
                )
            return arr - shift if shift else arr

        face_map: dict[str, npt.NDArray[np.int64]] = {}
        if zone_type is ZoneType.FEPOLYHEDRON:
            counts = section("node count per face", num_faces)
            if int(counts.sum()) != total_face_nodes:
                raise ValueError(
                    f"Poly zone node counts sum to {int(counts.sum())}, but "
                    f"TotalNumFaceNodes is {total_face_nodes}."
                )
            face_map["face_node_counts"] = counts
            face_map["face_nodes"] = section("face nodes", total_face_nodes, 1)
        else:
            face_map["face_nodes"] = section("face nodes", 2 * num_faces, 1)
        face_map["face_left_elements"] = section("left elements", num_faces, 1)
        face_map["face_right_elements"] = section("right elements", num_faces, 1)
        if num_boundary_faces:
            face_map["boundary_connection_counts"] = section(
                "boundary connection counts", num_boundary_faces
            )
            face_map["boundary_connection_elements"] = section(
                "boundary connection elements", num_boundary_connections, 1
            )
            face_map["boundary_connection_zones"] = section(
                "boundary connection zones", num_boundary_connections, 1
            )
        return face_map

    # -- Block readers -----------------------------------------------------------------

    @staticmethod
    def _read_float_block(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Read *n_values* floats from *tokens* into an intermediate float64 array.

        This is a pure text-parsing primitive. Callers cast the result to each
        variable's real dtype afterward, once individual per-variable arrays have been
        split out.

        Dispatches to :meth:`_read_float_block_fast`, a vectorized NumPy parser that
        handles the overwhelming majority of (well-formed) input in a handful of C-level
        calls regardless of block size, and falls back to :meth:`_read_float_block_slow`
        -- a tolerant, token-by-token parser -- only if the fast path can't make sense
        of what it read (a stray comment or non-numeric token inside the block, or a
        boundary reached before *n_values* values were collected). The fallback re-reads
        from *tokens* at the position the fast path started from, so the two never
        disagree about what was consumed.

        See Also:
            :meth:`_read_block_var_data`/:meth:`_read_point_var_data`.

        Examples:
            >>> arr = TecplotDatReader._read_float_block(tokens, 100)
        """
        if n_values <= 0:
            return np.empty(0, dtype=np.float64)

        marker = tokens.position()
        try:
            arr = TecplotDatReader._read_float_block_fast(tokens, n_values)
        except ValueError:
            arr = None
        if arr is not None:
            return arr

        tokens.seek(marker)
        return TecplotDatReader._read_float_block_slow(tokens, n_values)

    @staticmethod
    def _read_float_block_fast(
        tokens: _LineBuffer, n_values: int
    ) -> npt.NDArray | None:
        """Vectorized fast path for :meth:`_read_float_block`.

        Reads raw (unprocessed) lines in a handful of growing bulk grabs (sized from a
        one-line width probe, so a multi-million-value block costs a small constant
        number of Python-level calls). Passing ``sep`` here is the (fully supported)
        text-parsing mode of that function, not its deprecated binary-decoding mode.

        Returns ``None`` (never a short array) if a block boundary is hit before
        *n_values* values are collected, so the caller can unambiguously fall back to
        the slow, boundary-aware parser. A ``ValueError`` from ``numpy.fromstring`` is
        allowed to propagate to the caller for the same reason.

        Example:
            >>> arr = TecplotDatReader._read_float_block_fast(tokens, 100)

        """
        pieces: list[npt.NDArray] = []
        have = 0
        per_line: int | None = None
        for _ in range(_MAX_FAST_BLOCK_ITERATIONS):
            remaining = n_values - have
            if remaining <= 0:
                break
            # First grab is a single-line probe to measure the block's (typically
            # constant) values-per-line width
            n_lines = 1 if per_line is None else -(-remaining // per_line)
            batch = tokens.take_raw(n_lines)
            if not batch:
                return None  # ran out of input before reaching n_values
            piece = np.fromstring("".join(batch), dtype=np.float64, sep=" ")
            if per_line is None:
                per_line = max(piece.size, 1)
            pieces.append(piece)
            have += piece.size
        else:
            return None  # didn't converge; let the slow path sort it out
        arr = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return arr[:n_values] if arr.size >= n_values else None

    @staticmethod
    def _read_float_block_slow(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Tolerant, token-by-token fallback for :meth:`_read_float_block`.

        Reads one stripped line at a time, converting tokens one at a time and silently
        skipping any that aren't valid floats, stopping at a block boundary (see
        :func:`_is_float_block_boundary`) even if *n_values* hasn't been reached
        yet. Slow by design, but guarantees a result either way.

        Example:
            >>> arr = TecplotDatReader._read_float_block_slow(tokens, 100)

        """
        values: list[float] = []
        while len(values) < n_values and tokens.has_more():
            nxt = tokens.peek_stripped()
            if _is_float_block_boundary(nxt):
                break
            line = tokens.next_stripped()
            for tok in line.split():
                try:
                    values.append(float(tok))
                    if len(values) == n_values:
                        break
                except ValueError:
                    pass
        return np.array(values[:n_values], dtype=np.float64)

    @staticmethod
    def _read_block_var_data(
        tokens: _LineBuffer,
        num_vars: int,
        num_nodes: int,
        num_cells: int,
        var_locs: dict[int, ValueLocation],
        passive_set: set[int],
        share_map: dict[int, int],
        var_types: list[DataType],
    ) -> list[npt.NDArray | None]:
        """Read ``DATAPACKING=BLOCK`` variable data for one zone.

        One contiguous block of values is read per active variable, in dataset variable
        order. Each block is parsed as text into an intermediate ``float64`` array
        (``_read_float_block`` doesn't know or care about per-variable types), then cast
        to that variable's actual declared dtype (from ``DT=``, or the spec's documented
        SINGLE default) right here, where each variable's array is finalized.

        Reading one variable at a time (rather than all of them in a single combined
        call) is deliberate: :meth:`_read_float_block`'s vectorized fast path already
        costs only a handful of C-level calls per block regardless of size.

        Note:
            Passive and shared variables contribute a ``None`` placeholder to the
            returned list. The caller resolves shared variables against the zones parsed
            so far afterward, since this function only reads bytes off disk and has no
            zone list to resolve against.

        Example:
            >>> arrays = TecplotDatReader._read_block_var_data(
            ...     tokens, 3, 100, 80, {}, set(), {}, [DataType.FLOAT] * 3
            ... )

        """
        var_arrays: list[npt.NDArray | None] = [None] * num_vars
        for var_idx in range(num_vars):
            if var_idx in passive_set or var_idx in share_map:
                continue  # no data block for passive/shared variables
            loc = var_locs.get(var_idx, ValueLocation.NODAL)
            n_vals = num_cells if loc == ValueLocation.CELL_CENTERED else num_nodes
            raw = TecplotDatReader._read_float_block(tokens, n_vals)
            var_arrays[var_idx] = raw.astype(_DT_TO_DTYPE[var_types[var_idx]])
        return var_arrays

    @staticmethod
    def _read_point_var_data(
        tokens: _LineBuffer,
        num_vars: int,
        num_nodes: int,
        num_cells: int,
        var_locs: dict[int, ValueLocation],
        passive_set: set[int],
        share_map: dict[int, int],
        var_types: list[DataType],
    ) -> list[npt.NDArray | None]:
        """Read ``DATAPACKING=POINT`` variable data for one zone.

        The spec writes two interleaved sections:

            * **Nodal section** — ``num_nodes`` rows, one per node. Each row contains
              the values of all active nodal variables in dataset order.
            * **Cell-centered section** — ``num_cells`` rows, one per element. Each row
              contains the values of all active CC variables in dataset order.

        When all variables are nodal the cell-centered section is empty and is skipped
        automatically. Passive and shared variables are excluded from both sections and
        contribute a ``None`` placeholder to the returned list. Each column is cast to
        its variable's actual declared dtype (from ``DT=``, or the spec's documented
        SINGLE default) right where it's extracted from the interleaved rows.

        See Also:
            :meth:`_read_block_var_data`

        Example:
            >>> arrays = TecplotDatReader._read_point_var_data(
            ...     tokens, 3, 100, 80, {}, set(), {}, [DataType.FLOAT] * 3
            ... )

        """
        # Active variable indices, preserving dataset order.
        nodal_active: list[int] = [
            i
            for i in range(num_vars)
            if i not in passive_set
            and i not in share_map
            and var_locs.get(i, ValueLocation.NODAL) == ValueLocation.NODAL
        ]
        cc_active: list[int] = [
            i
            for i in range(num_vars)
            if i not in passive_set
            and i not in share_map
            and var_locs.get(i, ValueLocation.NODAL) == ValueLocation.CELL_CENTERED
        ]

        var_arrays: list[npt.NDArray | None] = [None] * num_vars

        # Nodal section: num_nodes rows × len(nodal_active) columns.
        n_nodal = len(nodal_active)
        if n_nodal > 0 and num_nodes > 0:
            flat = TecplotDatReader._read_float_block(tokens, num_nodes * n_nodal)
            # Row-major: flat[node * n_nodal + col] = value for that variable.
            matrix = flat.reshape(num_nodes, n_nodal)
            for col, var_idx in enumerate(nodal_active):
                var_arrays[var_idx] = np.ascontiguousarray(
                    matrix[:, col], dtype=_DT_TO_DTYPE[var_types[var_idx]]
                )

        # Cell-centred section: num_cells rows × len(cc_active) columns.
        n_cc = len(cc_active)
        if n_cc > 0 and num_cells > 0:
            flat = TecplotDatReader._read_float_block(tokens, num_cells * n_cc)
            matrix = flat.reshape(num_cells, n_cc)
            for col, var_idx in enumerate(cc_active):
                var_arrays[var_idx] = np.ascontiguousarray(
                    matrix[:, col], dtype=_DT_TO_DTYPE[var_types[var_idx]]
                )

        return var_arrays

    @staticmethod
    def _read_int_block(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Read exactly *n_values* integers from *tokens* into an int64 array.

        Used for connectivity (node-map) data, which can run to tens of millions of
        entries for large volume meshes. Follows the same vectorized-fast-path /
        tolerant-slow-path split as :meth:`_read_float_block`; see that method's
        docstring for the rationale.

        Examples:
            >>> arr = TecplotDatReader._read_int_block(tokens, 24)
        """
        if n_values <= 0:
            return np.empty(0, dtype=np.int64)

        marker = tokens.position()
        try:
            arr = TecplotDatReader._read_int_block_fast(tokens, n_values)
        except ValueError:
            arr = None
        if arr is not None:
            return arr

        tokens.seek(marker)
        return TecplotDatReader._read_int_block_slow(tokens, n_values)

    @staticmethod
    def _read_int_block_fast(tokens: _LineBuffer, n_values: int) -> npt.NDArray | None:
        """Vectorized fast path for :meth:`_read_int_block`.

        See :meth:`_read_float_block_fast`, which this mirrors exactly except for the
        target dtype.

        Example:
            >>> arr = TecplotDatReader._read_int_block_fast(tokens, 24)
        """
        pieces: list[npt.NDArray] = []
        have = 0
        per_line: int | None = None
        for _ in range(_MAX_FAST_BLOCK_ITERATIONS):
            remaining = n_values - have
            if remaining <= 0:
                break
            n_lines = 1 if per_line is None else -(-remaining // per_line)
            batch = tokens.take_raw(n_lines)
            if not batch:
                return None
            piece = np.fromstring("".join(batch), dtype=np.int64, sep=" ")
            if per_line is None:
                per_line = max(piece.size, 1)
            pieces.append(piece)
            have += piece.size
        else:
            return None
        arr = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return arr[:n_values] if arr.size >= n_values else None

    @staticmethod
    def _read_int_block_slow(tokens: _LineBuffer, n_values: int) -> npt.NDArray:
        """Tolerant, token-by-token fallback for :meth:`_read_int_block`.

        Example:
            >>> arr = TecplotDatReader._read_int_block_slow(tokens, 24)
        """
        values: list[int] = []
        while len(values) < n_values and tokens.has_more():
            nxt = tokens.peek_stripped()
            if _is_int_block_boundary(nxt):
                break
            line = tokens.next_stripped()
            for tok in line.split():
                try:
                    values.append(int(tok))
                    if len(values) == n_values:
                        break
                except ValueError:
                    pass
        return np.array(values[:n_values], dtype=np.int64)
