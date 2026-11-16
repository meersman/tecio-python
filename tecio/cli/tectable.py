r"""Convert a delimited text table to ordered Tecplot zones.

Simplify loading plain text tables into Tecplot with some additional tools that operate
on database-like text tables from a command-line interface.

Automatically parse numeric columns to ordered zone variables variables. While text
columns cannot be represented in data arrays, if every entry of the column is the same
within a zone, the text is stored as zone-level auxiliary data
``column_header=column_text`` and otherwise automatically skipped. Database parsing
tools allow families of data to be grouped my constant values similar to pivot tables
with ``-g``/``--group``. The group values are stored in each zone as auxiliary data, and
in its title. For data that is a function of two or more independent variables, optional
``-m``/``--mesh`` allow the uesr to specify columns containing one-dimensional
coordinate arrays to be converted to N-dimensional IJ(K) ordered arrays (such as a
surface of a coefficient over two angles). The user may filter data that is written to
file with ``--filter`` supporting all math comparison operators.

:Usage:

.. code:: bash

    tectable [-h] [-o PATH] [-f] [-d DELIMITER] [--header {auto,yes,no}]
             [-n NAMES] [--filter EXPR [--filter ...]] [-g COLUMN [-g ...]]
             [-r | --no-sort] [--keep-group] [-m COLUMN [-m ...]]
             [--precision {auto,single,double}] [--comment PREFIX] [PATH]

:Positional Arguments:
    ``PATH``
        Path to the delimited text table. Use ``-`` or leave it out to read standard
        input.

:Options:
    ``-o PATH``, ``--output PATH``
        Output file path. The extension determines the output format (either ``.szplt``,
        ``.plt``, or ``.dat``). Defaults to the input path with the extension
        ``.szplt``, or to ``table.szplt`` when reading standard input.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Files are not overwritten by
        default.

    ``-d DELIMITER``, ``--delimiter DELIMITER``
        Field separator: ``comma``, ``space`` (any run of spaces and tabs), ``tab``, or
        a single character such as ``;``. The default, ``auto``, picks ``comma`` if the
        first line holds a comma and ``space`` otherwise. Spaces around the fields are
        removed, and fields may be quoted with ``"``.

    ``--header {auto,yes,no}``
        Whether the first line holds the column names. With ``auto``, the first line is
        a header if a column that is numeric in the other rows is not numeric there.
        Without a header the variables are named ``V1``, ``V2``, and so on.

    ``-n NAMES``, ``--names NAMES``
        Comma-separated column names, replacing the header. A header line is then
        dropped only with ``--header yes``.

    ``--filter EXPR``
        Keep only the rows where ``COLUMN OP VALUE`` holds, with ``OP`` one of ``==``,
        ``!=``, ``<``, ``<=``, ``>`` and ``>=``. The column is named exactly as in the
        header (case-sensitive), so quote the expression in a shell, for example
        ``--filter 'Alpha!=0'``. Text columns support ``==`` and ``!=`` only.
        Repeatable: a row is kept if it satisfies all of the filters. Missing values
        (NaN) fail every comparison except ``!=``.

    ``-g COLUMN``, ``--group COLUMN``
        Split the rows into zones by the values of this column, named exactly as in
        the header (case-sensitive). Names with spaces are quoted in the shell, for
        example ``-g "Mach number"``. Repeatable: with several columns, rows are
        grouped by the first, then by the second within each of those, and so on.
        Zones are ordered by the group values, numerically for numeric columns and
        alphabetically for text columns. Rows keep their file order within a zone.

    ``-r``, ``--reverse``
        Order the zones by descending group values.

    ``--no-sort``
        Order the zones by the first appearance of each group in the table.

    ``--keep-group``
        Keep numeric group columns as variables. By default they are dropped, since
        each zone holds a single constant value that is stored as auxiliary data.

    ``-m COLUMN``, ``--mesh COLUMN``
        Write each zone as an IJ (two columns) or IJK (three columns) ordered zone
        instead of a list of points. The first column varies fastest. Every zone must
        hold each combination of the distinct values of these columns exactly once,
        which is a complete, not necessarily square, grid. Otherwise the command
        fails and names the zone. The columns must be numeric and not be group
        columns. They stay in the file as variables.

    ``--precision {auto,single,double}``
        Data type of the variables. With ``auto`` a column of whole numbers is stored
        as the smallest of ``uint8``, ``int16`` and ``int32`` that holds it. Any other
        column is stored in single precision if every value has at most 6 significant
        digits, which single precision holds as written, and in double precision
        otherwise. ``.szplt`` files choose per variable. A ``.dat`` file chooses the
        integer types per variable, and one precision for its other variables. A
        ``.plt`` file has no integer variables, and one precision for all of them. Use
        ``single`` or ``double`` to store every variable as a float of that precision.
        Default ``auto``.

    ``--comment PREFIX``
        Ignore text from this prefix to the end of a line. Default ``#``. An empty
        string turns it off.

:Returns:
    A Tecplot file with one ordered zone per group, or one zone without ``-g``. Exit
    code ``0`` on success and non-zero if the input cannot be read or parsed, a column
    cannot be resolved, a filter removes every row, a mesh is not complete, no numeric
    column remains, or the output file already exists and ``--force`` is not set.

Notes:
    A column is numeric if its first rows (up to 200,000) are all numbers. Empty fields
    and ``NA``, ``N/A``, ``null``, ``none`` and ``nan`` are read as NaN, and Fortran
    ``D`` exponents are accepted. Any other text in such a column later in the table
    is an error.

    Text values are matched exactly, apart from the removal of surrounding spaces.
    Auxiliary data names are the column names with other characters than letters,
    digits, ``_`` and ``.`` replaced by ``_``. Text columns that vary within some zone
    are skipped in that zone, with a note on standard error.

Examples:
    Convert a CSV file with a header row into ``results.szplt``::

        $ tectable results.csv

    Read from a pipe, picking the output format by extension::

        $ awk '$3 > 0' raw.txt | tectable -o filtered.plt

    Split rows into one zone per category, with the category as auxiliary data::

        $ tectable -g Category runs.csv

    Group by two columns, in descending order of the first::

        $ tectable -g Case -g Run -r runs.csv -o runs.szplt

    From an aerodynamic database, one zone for each Mach, Altitude and Beta, with the
    columns Alpha, CX, CY and so on as variables, and the three group values as
    auxiliary data of the zone::

        $ tectable -g Mach -g Altitude -g Beta --filter 'Alpha!=0' aero.csv

    Surfaces over Alpha and Beta, one for each Mach and Altitude::

        $ tectable -g Mach -g Altitude -m Alpha -m Beta aero.csv

    A semicolon separated table without a header::

        $ tectable -d ';' -n x,y,p scan.txt -o scan.dat

    Call directly from a Python session::

        from tecio.cli.tectable import main as tectable

        tectable(["-g", "Category", "table.csv", "-o", "table.szplt"])

    Without a file, ``table.szplt`` is written to the current directory::

        $ cat table.csv | tectable

See Also:
    :mod:`tecio.cli.teconvert` - Convert between Tecplot file formats.

"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import IO, Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from .. import open as tecio_open

# Rows parsed at a time, which also bounds the rows that decide the type of a column
_CHUNK_ROWS = 200_000
# Delimiter names mapped to the characters for np.loadtxt, None meaning any whitespace
_DELIMITERS: dict[str, str | None] = {"comma": ",", "space": None, "tab": "\t"}
_MISSING = ("", "na", "n/a", "null", "none")
_OUTPUT_EXTENSIONS = (".szplt", ".plt", ".dat")
_DEFAULT_OUTPUT = "table.szplt"
# Two-character operators come first so that "<=" is not read as "<"
_FILTER_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "==": np.equal,
    "!=": np.not_equal,
    "<=": np.less_equal,
    ">=": np.greater_equal,
    "<": np.less,
    ">": np.greater,
}
# Values of a column checked first when classifying it, so that a column that does not
# fit a narrow type is rejected at once
_SAMPLE_ROWS = 1024
# Integer types of Tecplot with their value range, smallest first
_INTEGER_TYPES = (
    (np.dtype(np.uint8), 0, 2**8 - 1),
    (np.dtype(np.int16), -(2**15), 2**15 - 1),
    (np.dtype(np.int32), -(2**31), 2**31 - 1),
)
# The .dat writer prints integers with 9 significant digits, which is exact below this
_DAT_INTEGER_LIMIT = 10**9 - 1

# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tectable",
        description=(
            # -|-------------------|---------------------------------------------|
            "Convert a delimited text table, from a file or standard input, to\n"
            "ordered Tecplot zones. Numeric columns become variables."
        ),
        epilog=(
            "Example usage:\n"
            "  Convert a CSV file with a header row\n"
            "    $ tectable results.csv\n"
            "  Read from a pipe\n"
            "    $ awk '$3 > 0' raw.txt | tectable -o filtered.plt\n"
            "  One zone per category, category stored as aux data\n"
            "    $ tectable -g Category runs.csv\n"
            "  Group by two columns, descending order\n"
            "    $ tectable -g Case -g Run -r runs.csv -o runs.szplt\n"
            "  Keep some rows, then make Alpha x Beta surfaces per Mach\n"
            "    $ tectable --filter 'Beta<3' -g Mach -m Alpha -m Beta a.csv\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="-",
        type=str,
        help="Input text table. Use - or omit to read standard input.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path (.szplt, .plt or .dat). Defaults to the input "
            "path with the extension .szplt, or table.szplt for standard input."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        type=str,
        default="auto",
        metavar="DELIMITER",
        help=(
            "Field separator: comma, space (runs of spaces and tabs), tab, or a "
            "single character. Default auto: comma if the first line has one, "
            "otherwise space."
        ),
    )
    parser.add_argument(
        "--header",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether the first line holds column names. Default auto.",
    )
    parser.add_argument(
        "-n",
        "--names",
        type=str,
        default=None,
        metavar="NAMES",
        help="Comma-separated column names, replacing the header.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="EXPR",
        dest="filters",
        help=(
            "Keep rows where 'COLUMN OP VALUE' holds, OP is one of == != < <= > >=. "
            "Repeatable, all must hold. Quote it, e.g. --filter 'Alpha!=0'."
        ),
    )
    parser.add_argument(
        "-g",
        "--group",
        action="append",
        default=[],
        metavar="COLUMN",
        help=(
            "Split rows into one zone per value of this column, named as in the "
            "header (case-sensitive). Repeatable for nested groups, first is "
            "the outermost."
        ),
    )
    order = parser.add_mutually_exclusive_group()
    order.add_argument(
        "-r",
        "--reverse",
        action="store_true",
        default=False,
        help="Order zones by descending group values.",
    )
    order.add_argument(
        "--no-sort",
        action="store_true",
        default=False,
        help="Order zones by first appearance of each group.",
    )
    parser.add_argument(
        "--keep-group",
        action="store_true",
        default=False,
        help="Keep numeric group columns as variables.",
    )
    parser.add_argument(
        "-m",
        "--mesh",
        action="append",
        default=[],
        metavar="COLUMN",
        help=(
            "Write zones as IJ(K) ordered zones over the distinct values of this "
            "numeric column. Repeatable up to 3 times, first varies fastest."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "single", "double"],
        default="auto",
        help=(
            "Variable type. auto stores whole numbers as integers and other "
            "columns in single precision if all values have at most 6 "
            "significant digits. Default auto."
        ),
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="#",
        metavar="PREFIX",
        help="Ignore text from this prefix to the end of a line. Default '#'.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Reading the table
# --------------------------------------------------------------------------------------


class _Table(NamedTuple):
    """A parsed table: numeric columns as values, text columns as coded labels."""

    names: list[str]
    numeric: dict[int, npt.NDArray[np.float64]]
    codes: dict[int, npt.NDArray[np.int32]]
    labels: dict[int, list[str]]
    num_rows: int


def _data_lines(stream: IO[str], comment: str | None) -> Iterator[str]:
    """Yield the lines that hold data, without blank and comment-only lines."""
    for raw in stream:
        text = raw.strip()
        if text and not (comment and text.startswith(comment)):
            yield text


def _resolve_delimiter(spec: str, first_line: str) -> str | None:
    """Return the ``np.loadtxt`` delimiter for *spec*, ``None`` for whitespace.

    Raises:
        ValueError: If *spec* is not a name or a single character.
    """
    name = spec.lower()
    if name == "auto":
        return "," if "," in first_line else None
    if name in _DELIMITERS:
        return _DELIMITERS[name]
    if len(spec) == 1:
        return None if spec == " " else spec
    raise ValueError(
        f"delimiter {spec!r} must be auto, comma, space, tab or a single character."
    )


def _split_fields(
    lines: Sequence[str], delimiter: str | None, comment: str | None
) -> npt.NDArray[np.str_]:
    """Split *lines* into fields.

    Returns:
        A ``(rows, columns)`` array of strings.

    Raises:
        ValueError: If the lines do not all have the same number of fields, or a quote
            is not closed on its line, which would join it with the next lines.
    """
    rows = np.loadtxt(
        lines,
        dtype=str,
        delimiter=delimiter,
        comments=comment or None,
        quotechar='"',
        ndmin=2,
    )
    if len(rows) != len(lines):
        raise ValueError("a quote is not closed on its line.")
    return rows


def _to_float(column: npt.NDArray[np.str_]) -> npt.NDArray[np.float64] | None:
    """Convert a column of strings to numbers, or return ``None`` if it is text.

    Missing-value words become NaN and a Fortran ``D`` exponent is read as ``E``. A
    column without a single number is text.

    Returns:
        The values, or ``None`` for a text column.
    """
    try:
        return column.astype(np.float64)
    except ValueError:
        pass
    low = np.char.lower(np.char.strip(column))
    low = np.where(np.isin(low, _MISSING), "nan", low)
    try:
        values = np.char.replace(low, "d", "e").astype(np.float64)
    except ValueError:
        return None
    return None if np.isnan(values).all() else values


def _load_floats(
    lines: Sequence[str], delimiter: str | None, comment: str | None, columns: int
) -> npt.NDArray[np.float64] | None:
    """Parse *lines* directly as numbers, which is much faster than via strings.

    Returns:
        A ``(rows, columns)`` array, or ``None`` if the lines hold anything but
        numbers, or do not have *columns* fields, or have an open quote. The caller
        then parses them again as text, which also describes the problem.
    """
    try:
        values = np.loadtxt(
            lines,
            dtype=np.float64,
            delimiter=delimiter,
            comments=comment or None,
            quotechar='"',
            ndmin=2,
        )
    except ValueError:
        return None
    return values if values.shape == (len(lines), columns) else None


def _factorize(
    column: npt.NDArray[np.str_], labels: list[str], index: dict[str, int]
) -> npt.NDArray[np.int32]:
    """Code the strings of a column as integers into *labels*, adding new strings.

    Returns:
        The index into *labels* of each string.
    """
    unique, inverse = np.unique(np.char.strip(column), return_inverse=True)
    lookup = np.empty(len(unique), dtype=np.int32)
    for k, text in enumerate(unique.tolist()):
        if text not in index:
            index[text] = len(labels)
            labels.append(text)
        lookup[k] = index[text]
    return lookup[inverse.reshape(-1)]


def _clean_names(fields: Sequence[str]) -> list[str]:
    """Make column names non-empty and unique.

    Returns:
        The cleaned names.
    """
    names: list[str] = []
    for j, field in enumerate(fields):
        name = field.strip() or f"V{j + 1}"
        base, count = name, 1
        while name in names:
            count += 1
            name = f"{base}_{count}"
        names.append(name)
    return names


def _ragged(
    lines: Sequence[str],
    delimiter: str | None,
    comment: str | None,
    num_columns: int,
    first_number: int,
    error: ValueError,
) -> ValueError:
    """Describe the first line that does not have *num_columns* fields.

    Args:
        lines:        The lines that were parsed together.
        delimiter:    Field delimiter for ``np.loadtxt``.
        comment:      Comment prefix, or ``None``.
        num_columns:  Expected number of fields.
        first_number: 1-based data line number of ``lines[0]``.
        error:        The error raised while parsing, used if no line is at fault.

    Returns:
        The error to raise, naming the line when one can be found.
    """
    for k, line in enumerate(lines):
        if _split_fields([line], delimiter, comment).shape[1] != num_columns:
            return ValueError(
                f"line {first_number + k}: expected {num_columns} fields."
            )
    return error


def _read_table(
    stream: IO[str],
    delimiter_spec: str,
    comment: str | None,
    header: str,
    names: list[str] | None,
) -> _Table:
    """Parse a delimited text table in chunks of rows.

    Args:
        stream:         Text stream to read.
        delimiter_spec: ``--delimiter`` value.
        comment:        Comment prefix, or ``None``.
        header:         ``auto``, ``yes`` or ``no``, whether the first line is a header.
        names:          Column names given by the user, or ``None``.

    Returns:
        The table. A column is numeric if its first chunk of rows is.

    Raises:
        ValueError: If the table is empty or ragged, a header cannot be used, or a
            column turns out not to be numeric after its first rows.
    """
    lines = _data_lines(stream, comment)
    first = next(lines, None)
    if first is None:
        raise ValueError("the input holds no data.")
    delimiter = _resolve_delimiter(delimiter_spec, first)
    pending = [first, *islice(lines, _CHUNK_ROWS)]
    try:
        rows = _split_fields(pending, delimiter, comment)
    except ValueError as exc:
        expected = _split_fields([first], delimiter, comment).shape[1]
        error = _ragged(pending, delimiter, comment, expected, 1, exc)
        raise error from exc
    num_columns = rows.shape[1]
    if names is not None and len(names) != num_columns:
        raise ValueError(
            f"--names has {len(names)} names but the table has {num_columns} columns."
        )

    first_fields, body = rows[0], rows[1:]
    body_values: list[npt.NDArray[np.float64] | None] = []
    if header == "auto":
        if names is None and len(body) > 0:
            body_values = [_to_float(body[:, j]) for j in range(num_columns)]
        is_header = any(
            v is not None and _to_float(first_fields[j : j + 1]) is None
            for j, v in enumerate(body_values)
        )
    else:
        is_header = header == "yes"
    if names is None:
        names = _clean_names(first_fields if is_header else [""] * num_columns)
    data = body if is_header else rows
    if len(data) == 0:
        raise ValueError("the input holds a header but no data rows.")

    numeric: dict[int, list[npt.NDArray[np.float64]]] = {}
    coded: dict[int, list[npt.NDArray[np.int32]]] = {}
    labels: dict[int, list[str]] = {}
    index: dict[int, dict[str, int]] = {}
    for j in range(num_columns):
        values = body_values[j] if is_header and body_values else None
        if values is None:
            values = _to_float(data[:, j])
        if values is not None:
            numeric[j] = [values]
        else:
            labels[j], index[j] = [], {}
            coded[j] = [_factorize(data[:, j], labels[j], index[j])]
    num_rows = len(data)
    line_number = len(rows) + 1

    while True:
        pending = list(islice(lines, _CHUNK_ROWS))
        if not pending:
            break
        if not coded:
            floats = _load_floats(pending, delimiter, comment, num_columns)
            if floats is not None:
                for j in range(num_columns):
                    numeric[j].append(floats[:, j])
                num_rows += len(floats)
                line_number += len(floats)
                continue
        try:
            rows = _split_fields(pending, delimiter, comment)
        except ValueError as exc:
            error = _ragged(pending, delimiter, comment, num_columns, line_number, exc)
            raise error from exc
        if rows.shape[1] != num_columns:
            error = _ragged(
                pending, delimiter, comment, num_columns, line_number, ValueError()
            )
            raise error
        for j in range(num_columns):
            if j in numeric:
                values = _to_float(rows[:, j])
                if values is None:
                    raise ValueError(
                        f"column '{names[j]}' is numeric at first but holds text "
                        f"near line {line_number}."
                    )
                numeric[j].append(values)
            else:
                coded[j].append(_factorize(rows[:, j], labels[j], index[j]))
        num_rows += len(rows)
        line_number += len(rows)

    return _Table(
        names=names,
        numeric={j: np.concatenate(v) for j, v in numeric.items()},
        codes={j: np.concatenate(v) for j, v in coded.items()},
        labels=labels,
        num_rows=num_rows,
    )


# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------


def _parse_filter(expr: str, names: list[str]) -> tuple[int, str, str]:
    """Split a filter such as ``Alpha>=2`` into its column, operator and value.

    The longest column name that starts the expression is used, so names may hold
    spaces and operator characters.

    Returns:
        The 0-based column, the operator and the value text.

    Raises:
        ValueError: If no column name is followed by an operator.
    """
    text = expr.strip()
    for j in sorted(range(len(names)), key=lambda k: -len(names[k])):
        if not text.startswith(names[j]):
            continue
        rest = text[len(names[j]) :].lstrip()
        for op in _FILTER_OPERATORS:
            if rest.startswith(op):
                return j, op, rest[len(op) :].strip()
    raise ValueError(
        f"cannot read filter '{expr}': expected COLUMN OP VALUE with COLUMN one of "
        f"{names} and OP one of {list(_FILTER_OPERATORS)}."
    )


def _apply_filters(table: _Table, filters: Sequence[str]) -> _Table:
    """Keep the rows of *table* that satisfy all *filters*.

    The columns are replaced one at a time in place, so the filtered table and a
    single original column are in memory together, not two copies of the table.

    Returns:
        The table with the remaining rows.

    Raises:
        ValueError: If a filter cannot be read or used with its column, or no row is
            left.
    """
    mask = np.ones(table.num_rows, dtype=bool)
    for expr in filters:
        j, op, value = _parse_filter(expr, table.names)
        compare = _FILTER_OPERATORS[op]
        if j in table.numeric:
            try:
                number = float(value)
            except ValueError:
                raise ValueError(
                    f"filter '{expr}': '{value}' is not a number, but column "
                    f"'{table.names[j]}' is numeric."
                ) from None
            mask &= compare(table.numeric[j], number)
        else:
            if op not in ("==", "!="):
                raise ValueError(
                    f"filter '{expr}': text column '{table.names[j]}' supports "
                    "only == and !=."
                )
            text = value[1:-1] if value[:1] == value[-1:] == '"' else value
            labels = table.labels[j]
            code = labels.index(text) if text in labels else -1
            mask &= compare(table.codes[j], code)
    kept = int(mask.sum())
    if kept == 0:
        raise ValueError("no row satisfies the filters.")
    if kept == table.num_rows:
        return table
    for column in (table.numeric, table.codes):
        for j in column:
            column[j] = column[j][mask]
    return table._replace(num_rows=kept)


# --------------------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------------------


class _Group(NamedTuple):
    """Rows of one group (``None`` for all rows) and the text of its group values."""

    rows: npt.NDArray[np.intp] | None
    key: list[str]


def _format_number(value: float) -> str:
    """Format a group value without a needless ``.0``.

    Returns:
        The text of the value.
    """
    if np.isfinite(value) and float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def _resolve_column(spec: str, names: list[str]) -> int:
    """Return the 0-based column named exactly *spec*.

    Raises:
        ValueError: If *spec* matches no column.
    """
    if spec in names:
        return names.index(spec)
    raise ValueError(f"column '{spec}' not found. Available: {names}")


def _split_groups(
    table: _Table, group_cols: list[int], *, reverse: bool, sort: bool
) -> list[_Group]:
    """Split the rows into groups of equal values in *group_cols*.

    Groups are in ascending order of their values, the first column varying slowest,
    or descending with *reverse*, or in order of first appearance without *sort*. Rows
    keep their table order within a group.

    Returns:
        The groups, each with its rows and the text of its group values.
    """
    if not group_cols:
        return [_Group(None, [])]
    codes: list[npt.NDArray[np.intp]] = []
    texts: list[list[str]] = []
    for j in group_cols:
        if j in table.numeric:
            unique, inverse = np.unique(table.numeric[j], return_inverse=True)
            codes.append(inverse.reshape(-1))
            texts.append([_format_number(v) for v in unique])
        else:
            ordered = np.argsort(np.array(table.labels[j]), kind="stable")
            rank = np.empty(len(ordered), dtype=np.intp)
            rank[ordered] = np.arange(len(ordered))
            codes.append(rank[table.codes[j]])
            texts.append([table.labels[j][k] for k in ordered])

    order = np.lexsort(codes[::-1])
    changed = np.zeros(len(order), dtype=bool)
    changed[0] = True
    for c in codes:
        ranked = c[order]
        changed[1:] |= ranked[1:] != ranked[:-1]
    starts = np.flatnonzero(changed)
    ends = np.append(starts[1:], len(order))
    groups = [
        _Group(order[s:e], [t[c[order[s]]] for t, c in zip(texts, codes, strict=True)])
        for s, e in zip(starts, ends, strict=True)
    ]
    if not sort:
        groups.sort(key=lambda g: int(cast(npt.NDArray[np.intp], g.rows)[0]))
    elif reverse:
        groups.reverse()
    return groups


# --------------------------------------------------------------------------------------
# Meshes and precision
# --------------------------------------------------------------------------------------


def _resolve_mesh(
    specs: Sequence[str], table: _Table, group_cols: list[int]
) -> list[int]:
    """Return the columns of the ``--mesh`` names.

    Raises:
        ValueError: If a name is unknown or repeated, there are more than 3, or a
            column is not numeric or is a group column.
    """
    cols = [_resolve_column(m, table.names) for m in specs]
    if len(cols) > 3 or len(set(cols)) != len(cols):
        raise ValueError("--mesh takes 1 to 3 different columns.")
    for j in cols:
        if j not in table.numeric or j in group_cols:
            raise ValueError(
                f"mesh column '{table.names[j]}' must be numeric and not a group."
            )
    return cols


def _mesh_layout(
    table: _Table, rows: npt.NDArray[np.intp] | None, mesh_cols: list[int], label: str
) -> tuple[npt.NDArray[np.intp], tuple[int, ...]]:
    """Order the rows of a zone as a complete grid over the *mesh_cols* values.

    Args:
        table:     The table.
        rows:      Table rows of the zone, or ``None`` for all rows.
        mesh_cols: Numeric columns that span the grid, the first varying fastest.
        label:     Text that names the zone in errors, or an empty string.

    Returns:
        The table rows in grid order, and the grid dimensions.

    Raises:
        ValueError: If a mesh column holds NaN, or the rows are not each combination
            of the distinct values exactly once.
    """
    count = table.num_rows if rows is None else len(rows)
    axes: list[npt.NDArray[np.intp]] = []
    dims: list[int] = []
    for j in mesh_cols:
        values = table.numeric[j] if rows is None else table.numeric[j][rows]
        if np.isnan(values).any():
            raise ValueError(
                f"{label}mesh column '{table.names[j]}' has missing values."
            )
        unique, inverse = np.unique(values, return_inverse=True)
        axes.append(inverse.reshape(-1))
        dims.append(len(unique))
    total = math.prod(dims)
    shape = " x ".join(str(d) for d in dims)
    names = " x ".join(table.names[j] for j in mesh_cols)
    if total != count:
        raise ValueError(
            f"{label}the rows are not a complete {names} grid: {count} rows for "
            f"{shape} = {total} points."
        )
    slot = np.ravel_multi_index(tuple(axes), tuple(dims), order="F")
    seen = np.zeros(total, dtype=bool)
    seen[slot] = True
    if not seen.all():
        raise ValueError(f"{label}the {names} grid has repeated points.")
    order = np.empty(total, dtype=np.intp)
    order[slot] = np.arange(total)
    return (order if rows is None else rows[order]), tuple(dims)


def _blocks(values: npt.NDArray[np.float64]) -> Iterator[npt.NDArray[np.float64]]:
    """Yield the first rows of *values*, then the rest in blocks of limited size.

    The small temporary arrays of a check stay small, and a check that fails on the
    first rows ends early.
    """
    step = max(_CHUNK_ROWS // 4, _SAMPLE_ROWS)
    edges = [0, min(len(values), _SAMPLE_ROWS), *range(step, len(values), step)]
    edges.append(len(values))
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        yield values[start:stop]


def _integer_dtype(values: npt.NDArray[np.float64], limit: int) -> np.dtype[Any] | None:
    """Return the smallest Tecplot integer type that holds every value, if any.

    Returns:
        ``uint8``, ``int16`` or ``int32``, or ``None`` if a value is not a whole number
        (NaN and infinity are not), or its magnitude is more than *limit*.
    """
    low = high = 0.0
    for block in _blocks(values):
        if not np.array_equal(block, np.rint(block)):
            return None
        if block.size:
            low, high = min(low, np.min(block)), max(high, np.max(block))
    if max(-low, high) > limit:
        return None
    return next((t for t, lo, hi in _INTEGER_TYPES if lo <= low and high <= hi), None)


def _fits_single(values: npt.NDArray[np.float64]) -> bool:
    """Tell if single precision holds every value as written in the table.

    Single precision keeps 6 significant decimal digits, so that is the test.

    Returns:
        ``True`` if every value has at most 6 significant digits.
    """
    for block in _blocks(values):
        magnitude = np.abs(block[np.isfinite(block)])
        magnitude = magnitude[magnitude != 0]
        if magnitude.size == 0:
            continue
        if magnitude.min() < 1e-30 or magnitude.max() > 1e30:
            return False
        # Scale to 6 digits before the decimal point, which must leave an integer
        scaled = magnitude * 10.0 ** (5 - np.floor(np.log10(magnitude)))
        if (np.abs(scaled - np.rint(scaled)) > 1e-8).any():
            return False
    return True


def _best_dtype(values: npt.NDArray[np.float64], limit: int) -> np.dtype[Any]:
    """Pick the narrowest type that keeps every value of a column as written.

    Returns:
        An integer type for whole numbers up to *limit*, else ``float32`` if single
        precision holds the values, else ``float64``.
    """
    integer = _integer_dtype(values, limit)
    if integer is not None:
        return integer
    return np.dtype(np.float32 if _fits_single(values) else np.float64)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def _aux_name(name: str) -> str:
    """Make a column name usable as an auxiliary data name.

    Returns:
        The name with characters other than letters, digits, ``_`` and ``.`` replaced.
    """
    return re.sub(r"[^A-Za-z0-9_.]", "_", name) or "_"


def main(argv: Sequence[str] | None = None) -> int:
    """Convert a delimited text table to ordered Tecplot zones.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.
    """
    args = _parse_args(argv)

    from_stdin = args.filename == "-"
    src = None if from_stdin else Path(args.filename)
    if src is not None and not src.is_file():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1
    if from_stdin and sys.stdin.isatty():
        print("Error: no input. Give a file or pipe a table in.", file=sys.stderr)
        return 1

    if args.output is not None:
        dst = Path(args.output)
    elif src is not None:
        dst = src.with_suffix(".szplt")
    else:
        dst = Path(_DEFAULT_OUTPUT)
    if dst.suffix.lower() not in _OUTPUT_EXTENSIONS:
        print(
            f"Error: output extension must be one of {_OUTPUT_EXTENSIONS}, "
            f"got '{dst.suffix}'.",
            file=sys.stderr,
        )
        return 1
    if src is not None and dst.resolve() == src.resolve():
        print("Error: the output would overwrite the input.", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    names = None if args.names is None else [n.strip() for n in args.names.split(",")]
    try:
        if src is None:
            table = _read_table(
                sys.stdin, args.delimiter, args.comment, args.header, names
            )
        else:
            with src.open(encoding="utf-8-sig") as stream:
                table = _read_table(
                    stream, args.delimiter, args.comment, args.header, names
                )
        read_rows = table.num_rows
        table = _apply_filters(table, args.filters)
        group_cols = list(
            dict.fromkeys(_resolve_column(g, table.names) for g in args.group)
        )
        mesh_cols = _resolve_mesh(args.mesh, table, group_cols)
        groups = _split_groups(
            table, group_cols, reverse=args.reverse, sort=not args.no_sort
        )
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    var_cols = [
        j for j in sorted(table.numeric) if args.keep_group or j not in group_cols
    ]
    if not var_cols:
        print("Error: no numeric column is left to write.", file=sys.stderr)
        return 1
    text_cols = [j for j in sorted(table.codes) if j not in group_cols]
    print(
        f"Read {read_rows} rows: {len(table.numeric)} numeric and "
        f"{len(table.codes)} text columns."
    )
    if args.filters:
        print(f"Kept {table.num_rows} of {read_rows} rows.")

    # Only .szplt files choose the type per variable, by the dtype of the data. A .dat
    # file takes integers per variable but one floating point precision for the rest.
    # A .plt file has one floating point precision for all variables.
    options: dict[str, str] = {}
    dtypes: dict[int, np.dtype[Any]] = {}
    suffix = dst.suffix.lower()
    if args.precision != "auto":
        options["precision"] = args.precision
    elif suffix == ".plt":
        fits = all(_fits_single(table.numeric[j]) for j in var_cols)
        options["precision"] = "single" if fits else "double"
    else:
        limit = _DAT_INTEGER_LIMIT if suffix == ".dat" else 2**31
        dtypes = {j: _best_dtype(table.numeric[j], limit) for j in var_cols}
        if suffix == ".dat":
            floats = {d for d in dtypes.values() if d.kind == "f"}
            options["precision"] = (
                "double" if np.dtype(np.float64) in floats else "single"
            )
        dtypes = {j: d for j, d in dtypes.items() if d != np.dtype(np.float64)}

    varying: set[str] = set()
    try:
        with tecio_open(
            str(dst),
            "w",
            title=src.stem if src is not None else "",
            variables=[table.names[j] for j in var_cols],
            **options,
        ) as writer:
            for group in groups:
                rows = group.rows
                title = ", ".join(
                    f"{table.names[j]}={text}"
                    for j, text in zip(group_cols, group.key, strict=True)
                )
                aux: dict[str, str] = {}
                for j, text in zip(group_cols, group.key, strict=True):
                    aux[_aux_name(table.names[j])] = text
                data: list[npt.NDArray[Any]]
                shape: tuple[int, ...] | None = None
                if mesh_cols:
                    label = f"zone '{title}': " if title else ""
                    index, shape = _mesh_layout(table, rows, mesh_cols, label)
                    data = [table.numeric[j][index] for j in var_cols]
                else:
                    data = [
                        table.numeric[j] if rows is None else table.numeric[j][rows]
                        for j in var_cols
                    ]
                for k, j in enumerate(var_cols):
                    if j in dtypes:
                        data[k] = data[k].astype(dtypes[j])
                    if shape is not None:
                        data[k] = data[k].reshape(shape, order="F")
                for j in text_cols:
                    codes = table.codes[j] if rows is None else table.codes[j][rows]
                    if (codes != codes[0]).any():
                        varying.add(table.names[j])
                    elif text := table.labels[j][codes[0]]:
                        aux[_aux_name(table.names[j])] = text
                writer.write_ordered_zone(
                    data=data, title=title or None, aux=aux or None
                )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    if varying:
        print(
            "Skipped text column(s) that vary within a zone: "
            + ", ".join(sorted(varying)),
            file=sys.stderr,
        )
    print(f"Wrote {len(groups)} zone(s) with {len(var_cols)} variable(s).")
    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
