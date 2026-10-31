r"""Merge zones from multiple Tecplot data files into a single output file.

Post-processing workflows commonly produce results distributed across multiple files.
``tecmerge`` collects all zones from an arbitrary number of input files into a single
output, reconciling variable lists across sources by taking their union and writing any
variable absent from a given source as passive. Input files may be specified explicitly
or via a quoted glob pattern and may be any mix of supported formats. When merging
time-step sequences, solution times can be assigned automatically from a start time and
either a fixed interval or an end time; each zone also gets a strand ID matching its
1-based position within its source file, so the same physical block (e.g. a wing zone
present in every timestep) shares one strand across the whole sequence and can be
animated as a single, continuous entity in the Tecplot GUI. Merged zones get
``SourceFile``/``SourceFileName`` aux data record of the source file and path. Grid data
can also be detected and shared across zones with matching structure instead of
duplicated in every merged zone.

:Usage:

.. code:: bash

    tecmerge [-h] -o PATH [-f] [--title STRING] [--assign-time-strands]
             [-s VALUE] [-d VALUE | -e VALUE] [--strand ID] [--strand-offset N]
             [--merge-grid [LIST]] FILE [FILE ...]

:Positional Arguments:
    ``FILE [FILE ...]``
        One or more input Tecplot files (``.plt``, ``.szplt``, or ``.dat``). Glob
        patterns are expanded by the tool — quote the pattern to prevent premature shell
        expansion (e.g.  ``"step_*.szplt"``). Files are merged in the order given or
        matched.

:Options:
    ``-o PATH``, ``--output PATH``
        Output file path. Required. The extension controls the output format:
        ``.szplt``, ``.plt``, or ``.dat``.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

    ``--title STRING``
        Dataset title to write to the output file. Defaults to the title of the first
        input file.

    ``--assign-time-strands``
        Assign evenly-spaced solution times to all zones, treating each input file as
        one time step. By default, each zone also gets a strand ID matching its 1-based
        position within its source file (the same block from every timestep shares a
        strand, letting the Tecplot GUI animate them together). Use ``--strand`` to
        force a single strand ID for every zone instead. Requires ``-s``/``--start``
        and either ``-d``/``--delta`` or ``-e``/``--end``.

    ``-s VALUE``, ``--start VALUE``
        Solution time of the first input file. Used with ``--assign-time-strands``.

    ``-d VALUE``, ``--delta VALUE``
        Constant time increment between successive input files. Mutually exclusive with
        ``-e``/``--end``.

    ``-e VALUE``, ``--end VALUE``
        Solution time of the last input file. The time step is computed as ``(end -
        start) / (N - 1)`` where ``N`` is the number of input files. Mutually exclusive
        with ``-d``/``--delta``.

    ``--strand INT``
        Override: force every zone to this single strand ID instead of the default
        per-zone-position assignment described under ``--assign-time-strands``. Has no
        effect without ``--assign-time-strands``.

    ``--strand-offset N``
        Added to every assigned strand ID (the per-zone-position default, or
        ``--strand`` if given). Lets separate ``tecmerge`` runs land on
        non-overlapping strand ranges, e.g. one run's zones get strands 1, 2, 3 and a
        second run with ``--strand-offset 100`` gets 101, 102, 103, so opening both
        outputs together in Tecplot doesn't auto-group zones from different merges onto
        the same strand. Has no effect without ``--assign-time-strands``. Defaults to
        ``0``.

    ``--merge-grid [LIST]``
        Detect and share grid data (coordinate variables, and for FE zones,
        connectivity) across zones with matching zone type and dimensions, instead of
        writing it fresh in every merged zone. Zones are compared by a ``(zone_type,
        num_nodes, num_elements)`` signature; the first zone with a given signature (in
        file, then zone, order) writes its grid fresh, and every later zone with a
        matching signature, whether from the same file or a different one, shares from
        it instead. This is not a rigorous check that the underlying data is actually
        identical, just that it plausibly could be, a false match (e.g. two genuinely
        different blocks that happen to share dimensions) would silently share the wrong
        grid. Polygon and polyhedron zones are matched on a hash of their full face map
        for simplicity. Poly zones with boundary face connections cannot be shared
        because boundary connections must be uniquely defined. If a zone only contains
        grid or passive variables, it cannot be merged because there would be no active
        data in the zone.

        Given with no value, grid variables are found automatically by name (``x``,
        ``X``, ``x-coordinate``, ``xgrid``, and similar common names for X/Y/Z). Given a
        comma-separated list of 1-based indices or exact names into the union variable
        list instead (e.g. ``--merge-grid x,y,z`` or ``--merge-grid 1,2,3``), only those
        variables are treated as the grid; indices and names cannot be mixed in the same
        list. Solution variables are never affected either way, only variables
        explicitly identified as the grid are ever shared.

:Returns:
    A new Tecplot file written to the output path containing all zones from every input
    file. Variables absent from a source file are written as passive. Exit code is ``0``
    on success and non-zero if any input file cannot be read, the output file already
    exists and ``--force`` is not set, or conflicting time-strand options are supplied.

Examples:
    Merge two files explicitly::

        $ tecmerge part1.szplt part2.szplt -o combined.szplt

    Merge a sequence matched by a glob pattern::

        $ tecmerg "results_*.szplt" -o combined.szplt

    Merge a time series, each zone getting a strand matching its position within its
    file (a wing zone present in every timestep shares one strand, so it animates as a
    single entity)::

        $ tecmerge --assign-time-strands -s 0.0 -d 0.1 \\
                   "step_*.szplt" -o transient.szplt

    Same, but force every zone onto a single strand instead::

        $ tecmerge --assign-time-strands -s 0.0 -d 0.1 --strand 1 \\
                   "step_*.szplt" -o transient.szplt

    Merge two separate time series so both can be opened together without the
    Tecplot GUI auto-grouping their zones onto shared strands::

        $ tecmerge --assign-time-strands -s 0.0 -d 0.1 \\
                   "groupA_*.szplt" -o groupA.szplt
        $ tecmerge --assign-time-strands -s 0.0 -d 0.1 --strand-offset 100 \\
                   "groupB_*.szplt" -o groupB.szplt

    Merge a multiblock time series, sharing each block's grid across timesteps
    instead of duplicating it::

        $ tecmerge --assign-time-strands -s 0.0 -d 0.1 --merge-grid \\
                   "step_*.szplt" -o transient.szplt

    Same, but only "x" and "y" are grid variables (a 2-D case with no "z")::

        $ tecmerge --merge-grid x,y "step_*.szplt" -o transient.szplt

    Call directly from a Python session::

        import tecio.cli.tecmerge.main as tecmerge

        tecmerge(["part1.szplt", "part2.szplt", "--output", "combined.szplt"])

See Also:
    * :mod:`tecio.cli.tecextract` - Extract a zone/variable subset from a single file —
      the inverse of merging.
    * :mod:`tecio.cli.tecsplit` - Split a file into separate grid and solution files.
    * :mod:`tecio.cli.tecslice` - Extract planar slices from volumetric zone data.

"""

from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import (
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotPolyZoneReader,
    TecplotReader,
    TecplotSzlWriter,
    TecplotWriter,
    TecplotZoneReader,
)
from .. import open as tecio_open
from .._writer import poly_face_kwargs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecmerge",
        description=(
            # -|--------------------|---------------------------------------------|
            "Merge zones from multiple Tecplot files into a single output file.\n"
            "Variables not present in a source file are written as passive."
        ),
        epilog=(
            # -|--------------------|---------------------------------------------|
            "Example usage:\n"
            "  Merge explicit files\n"
            "    $ tecmerge -o combined.szplt part1.szplt part2.szplt\n"
            "  Merge via glob\n"
            '    $ tecmerge -o combined.szplt "results_*.szplt"\n'
            "  Assign time/strand metadata\n"
            "    $ tecmerge --assign-time-strands -s 0.0 -d 0.1 \\\n"
            '               -o transient.szplt "step_*.szplt"\n'
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )

    # Input files
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        metavar="FILE",
        help=(
            "Input Tecplot files. Glob patterns are expanded (quote the "
            "pattern to prevent shell expansion). Files are merged in the "
            "order given / matched."
        ),
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Output file path. The extension controls the output format "
            "(.szplt, .plt, .dat)."
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
        "--title",
        type=str,
        default=None,
        metavar="STRING",
        help=(
            "Dataset title for the output file. Defaults to the title of "
            "the first input file."
        ),
    )

    # Time strand assignment
    ts = parser.add_argument_group("time/strand assignment (--assign-time-strands)")
    ts.add_argument(
        "--assign-time-strands",
        action="store_true",
        default=False,
        dest="assign_ts",
        help=(
            "Assign evenly-spaced solution times to all zones, treating each input "
            "file as one time step. By default strand IDs are automatically set; use "
            "--strand to force a single strand ID for every zone instead. Requires "
            "-s/--start and one of -d/--delta or -e/--end."
        ),
    )
    ts.add_argument(
        "-s",
        "--start",
        type=float,
        default=None,
        metavar="VALUE",
        help="Solution time of the first input file.",
    )
    # -d/--delta and -e/--end are mutually exclusive -- either specifies the
    # spacing, the other specifies the endpoint. Providing both over-constrains
    # the problem and is therefore disallowed at the parser level.
    step_group = ts.add_mutually_exclusive_group()
    step_group.add_argument(
        "-d",
        "--delta",
        type=float,
        default=None,
        metavar="VALUE",
        help="Constant time step between successive input files.",
    )
    step_group.add_argument(
        "-e",
        "--end",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "Solution time of the last input file. The step is computed as "
            "(end - start) / (N - 1) where N is the number of input files."
        ),
    )
    ts.add_argument(
        "--strand",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Override: force every zone to this single strand ID instead "
            "of the default per-zone-position assignment (see "
            "--assign-time-strands)."
        ),
    )
    ts.add_argument(
        "--strand-offset",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Added to every assigned strand ID (the default per-zone-position "
            "value, or --strand if given). Lets separate tecmerge runs land on "
            "non-overlapping strand ranges (e.g. --strand-offset 100 on a second "
            "run) so the Tecplot GUI doesn't auto-group zones from different "
            "merges when the outputs are opened together. Has no effect without "
            "--assign-time-strands. Defaults to 0 (no change)."
        ),
    )

    # Grid sharing
    parser.add_argument(
        "--merge-grid",
        nargs="?",
        const="__auto__",
        default=None,
        type=str,
        metavar="LIST",
        help=(
            "Detect and share the grid (coordinate variables, and connectivity for FE "
            "zones) with matching zone type and dimensions. To manually specify grid "
            "coordinates provide a comma-separated list of 1-based indices or exact "
            "variable names (e.g.  -merge-grid x,y,z or -merge-grid 1,2,3)."
        ),
    )

    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

# Guesses of grid variable names automatic grid variable detection for merging
_AXIS_SYNONYMS: dict[str, frozenset[str]] = {
    "x": frozenset({
        "x",
        "x-coordinate",
        "x coordinate",
        "coordinate x",
        "coord-x",
        "coordx",
        "xcoord",
        "x_coord",
        "xgrid",
        "x-grid",
        "x_grid",
    }),
    "y": frozenset({
        "y",
        "y-coordinate",
        "y coordinate",
        "coordinate y",
        "coord-y",
        "coordy",
        "ycoord",
        "y_coord",
        "ygrid",
        "y-grid",
        "y_grid",
    }),
    "z": frozenset({
        "z",
        "z-coordinate",
        "z coordinate",
        "coordinate z",
        "coord-z",
        "coordz",
        "zcoord",
        "z_coord",
        "zgrid",
        "z-grid",
        "z_grid",
    }),
}


def _autodetect_grid_variables(names: list[str]) -> list[int]:
    """Return the 1-based indices in *names* that look like X/Y/Z coordinates.

    Matched by exact name (case-insensitive, trimmed) against
    :data:`_AXIS_SYNONYMS`, checked in X, Y, Z order; an axis with no match among
    *names* is simply omitted rather than guessed at positionally, unlike the
    ParaView plugin's coordinate resolution, an incorrect guess here would silently
    mishandle real data instead of just looking odd on screen, so this only
    returns indices it's actually confident in.

    Example:
        >>> _autodetect_grid_variables(["x-grid", "y", "pressure"])
        [1, 2]
    """
    indices: list[int] = []
    for axis in ("x", "y", "z"):
        synonyms = _AXIS_SYNONYMS[axis]
        for i, name in enumerate(names, start=1):
            if name.strip().lower() in synonyms:
                indices.append(i)
                break
    return indices


def _parse_index_or_name_list(value: str) -> list[int | str]:
    """Parse a comma-separated string of 1-based integers and/or variable names.

    Each token is parsed as an integer where possible; anything else is kept as a
    name string, resolved against the union variable list once it's known.

    Args:
        value: String like ``"1,2,3"`` or ``"x,y,z"``.

    Returns:
        List of ``int`` (1-based index) and/or ``str`` (variable name) tokens, in
        the order given.

    Example:
        >>> _parse_index_or_name_list("x,y,z")
        ['x', 'y', 'z']
    """
    tokens: list[int | str] = []
    for raw in value.split(","):
        token = raw.strip()
        try:
            tokens.append(int(token))
        except ValueError:
            tokens.append(token)
    return tokens


def _resolve_grid_variable_indices(
    explicit: str | None, union_vars: list[str]
) -> set[int] | None:
    """Resolve ``--merge-grid``'s value into a set of 1-based union indices.

    Args:
        explicit: ``None`` if ``--merge-grid`` wasn't given at all (the feature is off);
            the sentinel ``"__auto__"`` if given with no value (detect by name, see
            :func:`_autodetect_grid_variables`); otherwise a comma-separated list of
            indices and/or names, all of one kind, not mixed (matching
            ``-v``/``--variables`` elsewhere in this project).
        union_vars: The merge's full union variable list.

    Returns:
        1-based indices into *union_vars* eligible for grid sharing, or ``None`` if
        ``--merge-grid`` wasn't given at all. An empty set (rather than ``None``) means
        the feature is on but nothing was found/specified to share, --merge-grid then
        has no effect.

    Raises:
        ValueError: If an explicit index is out of range, an explicit name isn't in
            *union_vars*, or indices and names are mixed in the same list.
    """
    if explicit is None:
        return None
    if explicit == "__auto__":
        return set(_autodetect_grid_variables(union_vars))

    tokens = _parse_index_or_name_list(explicit)
    has_int = any(isinstance(t, int) for t in tokens)
    has_name = any(isinstance(t, str) for t in tokens)
    if has_int and has_name:
        raise ValueError(
            "--merge-grid indices and names cannot be mixed in the same list; "
            f"use all indices or all names, got: {explicit!r}."
        )

    resolved: set[int] = set()
    for token in tokens:
        if isinstance(token, int):
            if token < 1 or token > len(union_vars):
                raise ValueError(
                    f"--merge-grid variable index {token} out of range "
                    f"[1, {len(union_vars)}]."
                )
            resolved.add(token)
        else:
            try:
                resolved.add(union_vars.index(token) + 1)
            except ValueError:
                raise ValueError(
                    f"--merge-grid variable name {token!r} not found; available "
                    f"names: {', '.join(union_vars)}."
                ) from None
    return resolved


def _expand_inputs(patterns: list[str]) -> list[Path]:
    """Expand a list of file paths / glob patterns to a sorted list of Paths.

    Args:
        patterns: Strings that may be literal file paths or glob patterns.

    Returns:
        Deduplicated, ordered list of :class:`Path` objects.

    Raises:
        SystemExit: If a pattern matches no files.

    """
    seen: set[Path] = set()
    result: list[Path] = []

    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            # Try treating as a literal path even if glob found nothing.
            p = Path(pattern)
            if p.exists():
                matches = [str(p)]
            else:
                raise FileNotFoundError(f"no files matched pattern: {pattern!r}")
        for m in matches:
            p = Path(m).resolve()
            if p not in seen:
                seen.add(p)
                result.append(p)

    return result


def _build_var_union(
    readers: list[TecplotReader],
) -> tuple[list[str], list[list[int | None]]]:
    """Compute the union variable list and per-reader index maps.

    Args:
        readers: List of open reader instances.

    Returns:
        A 2-tuple of:
        - ``union_vars``: Ordered list of all unique variable names
          (preserving first-seen order).
        - ``index_maps``: For each reader, a list of length
          ``len(union_vars)`` where each entry is either the 0-based
          local variable index for that reader, or ``None`` if the
          reader does not have that variable.

    """
    union_vars: list[str] = []
    union_set: dict[str, int] = {}  # name -> position in union_vars

    for reader in readers:
        for name in reader.variables:
            if name not in union_set:
                union_set[name] = len(union_vars)
                union_vars.append(name)

    index_maps: list[list[int | None]] = []
    for reader in readers:
        local_names = reader.variables
        local_map: dict[str, int] = {n: i for i, n in enumerate(local_names)}
        row: list[int | None] = [local_map.get(uname) for uname in union_vars]
        index_maps.append(row)

    return union_vars, index_maps


def _has_own_data(
    zone: TecplotZoneReader,
    local_index_map: list[int | None],
    grid_var_indices: set[int] | None,
) -> bool:
    """Return whether *zone* has a non-grid variable that is neither shared nor passive.

    Without one, sharing the grid leaves nothing to write and the zone is dropped.
    """
    for union_i, local_idx in enumerate(local_index_map, start=1):
        if local_idx is None or (grid_var_indices and union_i in grid_var_indices):
            continue
        var = zone.variables[local_idx]
        if not var.is_passive() and var.shared_zone is None:
            return True
    return False


def _write_zone(
    writer: TecplotWriter,
    zone: TecplotZoneReader,
    union_vars: list[str],
    local_index_map: list[int | None],
    solution_time: float | None,
    strand_id: int | None,
    zone_index_map: dict[int, int],
    source_path: Path,
    grid_var_indices: set[int] | None,
    grid_reference: int | None,
) -> None:
    """Write one zone to *writer* using the reconciled variable list.

    Variables present in *zone* are copied verbatim. Variables absent from
    *zone* (``None`` entries in *local_index_map*) are written as passive.

    Args:
        writer:           Open writer instance.
        zone:             Source zone reader.
        union_vars:       Full union variable name list.
        local_index_map:  Map from union index -> local 0-based var index (``None`` =
                          not present in this file).
        solution_time:    Override solution time, or ``None`` to keep original.
        strand_id:        Override strand ID, or ``None`` to keep original.
        zone_index_map:   Map 0-based source zone to 0-based output zone index for
                          variable and connectivity sharing.
        source_path:      Input file this zone came from, recorded as zone-level aux
                          data (``SourceFile``, ``SourceFileName``) on every merged
                          zone.
        grid_var_indices: 1-based union indices of the ``--merge-grid`` grid variables,
                          or ``None`` if the feature is off.
        grid_reference:   1-based output zone index to share this zone's grid variables
                          (and, for FE zones, connectivity) from, if an earlier zone
                          with a matching (zone_type, num_nodes, num_elements) signature
                          was already written, else ``None`` (this zone's grid is
                          written fresh; the caller is responsible for registering its
                          signature afterward). Ignored entirely when *grid_var_indices*
                          is ``None``.
    """
    zt = zone.zone_type

    data: list[np.ndarray] = []
    value_locations: list[Any] = []
    passive_vars: set[str] = set()
    var_sharing: dict[str, int] = {}

    for union_i, local_idx in enumerate(local_index_map, start=1):
        name = union_vars[union_i - 1]
        is_grid_var = grid_var_indices is not None and union_i in grid_var_indices

        if is_grid_var and grid_reference is not None:
            # An earlier zone with the same zone_type/num_nodes/num_elements already
            # wrote this grid so share rather than duplicate
            var_sharing[name] = grid_reference
            continue

        if local_idx is None:
            # Variable not in this file -- mark passive.
            passive_vars.add(name)
            continue

        var = zone.variables[local_idx]

        if var.is_passive():
            passive_vars.add(name)
            continue

        sv = var.shared_zone
        remapped = zone_index_map.get(sv) if sv is not None else None

        if remapped is not None:
            # Source zone was already written earlier in this file -> preserve sharing
            # relationship
            var_sharing[name] = remapped
            continue

        arr = var.values
        if arr is None or arr.size == 0:
            passive_vars.add(name)
        else:
            data.append(arr)
            value_locations.append(var.value_location)

    zone_aux: dict[str, str] = {}
    if len(zone.auxdata) > 0:
        zone_aux = dict(zone.auxdata.items())
    # Always added, source provenance is cheap to record and easy to ignore in the
    # Tecplot GUI if unwanted
    zone_aux["SourceFile"] = str(source_path)
    zone_aux["SourceFileName"] = source_path.name

    s_time = solution_time if solution_time is not None else zone.solution_time
    s_id = strand_id if strand_id is not None else zone.strand_id

    common_kw: dict[str, Any] = {
        "title": zone.title,
        "value_locations": value_locations,
        "passive_vars": passive_vars,
        "var_sharing": var_sharing,
        "solution_time": s_time,
        "strand_id": s_id,
        "aux": zone_aux,
    }

    if isinstance(zone, TecplotOrderedZoneReader):
        writer.write_ordered_zone(data=data, **common_kw)
    elif isinstance(zone, TecplotPolyZoneReader):
        if grid_var_indices is not None and grid_reference is not None:
            # The reference zone has an identical face map (checked by hash).
            con_remapped: int | None = grid_reference
        else:
            con_src = zone.shared_connectivity
            con_remapped = zone_index_map.get(con_src) if con_src is not None else None
        # Boundary connections follow the output numbering of this file's zones, a zone
        # may also be connected to itself.
        zone_map = {**zone_index_map, zone.zone_index: writer.current_zone + 1}
        writer.write_poly_zone(
            zone_type=zt,
            data=data,
            con_sharing=con_remapped,
            **({} if con_remapped is not None else poly_face_kwargs(zone, zone_map)),
            **common_kw,
        )
    elif isinstance(zone, TecplotFEZoneReader):
        if grid_var_indices is not None and grid_reference is not None:
            # Share connectivity from the same reference zone as the grid variables, a
            # matching (zone_type, num_nodes, num_elements) signature implies the same
            # mesh, node map included.
            con_remapped = grid_reference
        else:
            con_src = zone.shared_connectivity
            con_remapped = zone_index_map.get(con_src) if con_src is not None else None
        fe_kw = common_kw.copy()

        # Face-neighbor connections
        if zone.face_neighbor_mode is not None and not isinstance(
            writer, TecplotSzlWriter
        ):
            fe_kw["face_neighbors"] = zone.get_face_connections()
            fe_kw["face_neighbor_mode"] = zone.face_neighbor_mode
        writer.write_fe_zone(
            zone_type=zt,
            data=data,
            node_map=None if con_remapped is not None else zone.node_map,
            con_sharing=con_remapped,
            **fe_kw,
        )
    else:
        raise NotImplementedError(
            f"Zone '{zone.title}' is neither an ordered, classic FE nor poly "
            "zone; merging is not supported for it."
        )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Merge zones from multiple Tecplot files into one.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    # Validate time/strand options
    if args.assign_ts:
        if args.start is None:
            print("Error: --assign-time-strands requires -s/--start.", file=sys.stderr)
            return 1
        if args.delta is None and args.end is None:
            print(
                "Error: --assign-time-strands requires either -d/--delta or -e/--end.",
                file=sys.stderr,
            )
            return 1
    # Expand input globs
    try:
        input_paths = _expand_inputs(args.files)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not input_paths:
        print("Error: no input files found.", file=sys.stderr)
        return 1

    dst = Path(args.output)
    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    n_files = len(input_paths)
    print(f"Merging {n_files} file(s)  ->  {dst}")
    for p in input_paths:
        print(f"  {p}")

    # Build time schedule
    times: list[float] | None = None
    if args.assign_ts:
        if args.delta is not None:
            times = [args.start + i * args.delta for i in range(n_files)]
        else:
            if n_files == 1:
                times = [args.start]
            else:
                step = (args.end - args.start) / (n_files - 1)
                times = [args.start + i * step for i in range(n_files)]
        strand_desc = (
            f"fixed strand {args.strand} for all zones"
            if args.strand is not None
            else "per-zone-position strand (1, 2, 3, ... by zone order in each file)"
        )
        if args.strand_offset:
            strand_desc += f", offset by {args.strand_offset:+d}"
        print(f"\nTime assignment: {strand_desc}, times={[f'{t:.6g}' for t in times]}")

    try:
        # Open all readers
        readers: list[TecplotReader] = [tecio_open(str(p), "r") for p in input_paths]

        # Build union variable list
        union_vars, index_maps = _build_var_union(readers)
        n_union = len(union_vars)

        print(f"\nUnion variable list ({n_union}): {union_vars}")

        try:
            grid_var_indices = _resolve_grid_variable_indices(
                args.merge_grid, union_vars
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if grid_var_indices is not None:
            grid_names = [union_vars[i - 1] for i in sorted(grid_var_indices)]
            print(f"Grid sharing enabled for variables: {grid_names}")

        # Report any variables that will be passive in some files.
        for fi, (_reader, imap) in enumerate(zip(readers, index_maps, strict=False)):
            missing = [union_vars[ui] for ui, li in enumerate(imap) if li is None]
            if missing:
                print(
                    f"  {input_paths[fi].name}: variables set passive "
                    f"(not in file): {missing}"
                )

        # Resolve output title
        out_title = args.title if args.title is not None else readers[0].title

        # Open writer and stream all zones
        with tecio_open(
            str(dst),
            "w",
            title=out_title,
            variables=union_vars,
            file_type=readers[0].file_type,
        ) as writer:
            # Dataset aux data from first file only.
            if len(readers[0].auxdata) > 0:
                writer.add_auxdataset_dict(dict(readers[0].auxdata.items()))

            # Variable aux data: first file wins for variables it has.
            auxvar: dict[int, dict[str, str]] = {}
            for ui, local_idx in enumerate(index_maps[0]):
                if local_idx is None:
                    continue
                var_aux = readers[0].get_var_auxdata(local_idx)
                if len(var_aux) > 0:
                    auxvar[ui] = dict(var_aux.items())
            if auxvar:
                writer.add_auxvar_dict(auxvar)

            total_zones = 0
            # Persists sharing registry across every input file for merge grid option
            signature_registry: dict[tuple[Any, ...], int] = {}
            for fi, (reader, imap) in enumerate(zip(readers, index_maps, strict=False)):
                sol_time = times[fi] if times is not None else None

                # Sharing is always file-local (a zone can only share from another zone
                # in the same physical source file), so this map is reset fresh for
                # every input file rather than accumulated across the whole merge
                zone_index_map: dict[int, int] = {}

                for zi, zone in enumerate(reader.zones):
                    zone_num = zi + 1
                    zt = zone.zone_type

                    # Automatically set all. --strand overrides this with a single,
                    # fixed strand ID for every zone instead. --strand-offset then
                    # shifts whichever value was chosen.
                    if times is None:
                        s_id = None
                    elif args.strand is not None:
                        s_id = args.strand + args.strand_offset
                    else:
                        s_id = zone_num + args.strand_offset

                    grid_reference: int | None = None
                    signature: tuple[Any, ...] | None = None
                    if grid_var_indices is not None:
                        if isinstance(
                            zone, (TecplotOrderedZoneReader, TecplotFEZoneReader)
                        ):
                            signature = (zt, zone.num_nodes, zone.num_elements)
                        elif (
                            isinstance(zone, TecplotPolyZoneReader)
                            and not zone.num_boundary_faces
                        ):
                            # A poly grid is matched on its face map, not only its
                            # counts. Zones with boundary faces cannot be sharing
                            # sources, so they always write their own grid.
                            signature = (zt, zone._face_map_digest())
                        if signature is not None:
                            grid_reference = signature_registry.get(signature)

                    # A zone that would hold no data of its own is skipped by the
                    # writer, so write its grid fresh rather than lose the zone.
                    if grid_reference is not None and not _has_own_data(
                        zone, imap, grid_var_indices
                    ):
                        grid_reference = None

                    zones_before = writer.current_zone
                    _write_zone(
                        writer=writer,
                        zone=zone,
                        union_vars=union_vars,
                        local_index_map=imap,
                        solution_time=sol_time,
                        strand_id=s_id,
                        zone_index_map=zone_index_map,
                        source_path=input_paths[fi],
                        grid_var_indices=grid_var_indices,
                        grid_reference=grid_reference,
                    )
                    if writer.current_zone == zones_before:
                        continue  # skipped as empty, nothing to map or register
                    zone_index_map[zi] = writer.current_zone
                    if signature is not None and grid_reference is None:
                        # First zone with this signature should write grid, every later
                        # zone with a matching signature shares from it
                        signature_registry[signature] = writer.current_zone
                    total_zones += 1

        # Close all readers
        for reader in readers:
            reader.__exit__(None, None, None)

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"\nDone. {total_zones} zone(s) written to: {dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
