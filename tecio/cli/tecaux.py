r"""Add, remove, or export dataset, zone, or variable level auxiliary data.

The Tecplot auxiliary data mechanism attaches arbitrary ``name=value`` metadata to a
dataset, a zone, or a variable (solver name, run date, units, a description, or any
other annotation that doesn't belong in the numerical data itself). Managing this after
the fact ordinarily means writing a one-off script against the TecIO API.  ``tecaux``
does this from the command line, in a single read/write pass, in three mutually
exclusive modes:

* By default (no ``--strip``/``--export-json``): merges new auxiliary entries (from
  ``-d``/``-z``/``-v`` and/or ``-j``) into a copy of the input file. Every zone,
  variable, and existing sharing relationship is preserved exactly; an existing key with
  the same name is overwritten.
* ``--strip`` removes every auxiliary entry at all three levels and writes the result to
  a new file, still preserving sharing.
* ``--export-json`` writes every *existing* auxiliary entry to a JSON file, in the same
  format ``-j`` reads back -- without touching the source file at all, unless combined
  with ``--strip``.

:Usage:

.. code:: bash

    tecaux [-h] [-d KEY=VALUE] ... [-z INDEX KEY=VALUE] ...
           [-v INDEX_OR_NAME KEY=VALUE] ... [-j PATH] [-s] [--export-json]
           [-o PATH] [-f] PATH

:Positional Arguments:
    ``PATH``
        Path to the input Tecplot file (``.plt``, ``.szplt``, or ``.dat``).

:Options:
    ``-d KEY=VALUE``, ``--data KEY=VALUE``
        A ``name=value`` pair to set as dataset-level auxiliary data. Repeat the flag
        for multiple pairs.

    ``-z INDEX KEY=VALUE``, ``--zone INDEX KEY=VALUE``
        A ``name=value`` pair to set as zone-level auxiliary data on the 0-based zone
        ``INDEX``, or on every zone if ``INDEX`` is the literal word ``all``. Repeat the
        flag for multiple pairs and/or multiple zones; each occurrence takes exactly one
        zone and one pair, so ``-z 0 A=1 -z 0 B=2`` sets both ``A`` and ``B`` on zone 0.

    ``-v INDEX_OR_NAME KEY=VALUE``, ``--variable INDEX_OR_NAME KEY=VALUE``
        A ``name=value`` pair to set as variable-level auxiliary data on the variable
        given by a 0-based index or a name (case-insensitive), r on every variable
        if the target is the literal word ``all``. Repeatable, same as ``-z``.

    ``-j PATH``, ``--json PATH``
        Load bulk auxiliary data from a JSON file (see format below). Applied before
        any ``-d``/``-z``/``-v`` flags, which take precedence on a key collision.

    ``-s``, ``--strip``
        Remove all auxiliary data (dataset, zone, and variable levels) and write the
        result to ``<stem>_no_aux<ext>`` (or ``-o``'s path). Mutually exclusive with
        ``-d``/``-z``/``-v``/``-j``. Combine with ``--export-json`` to keep a JSON copy
        of what was removed.

    ``--export-json``
        Write every existing auxiliary entry to ``<stem>_aux.json``, in the same
        ``AUXDATASET``/``AUXZONE``/``AUXVAR`` format ``-j`` reads, without modifying
        the source file, unless combined with ``--strip``. Mutually exclusive with
        ``-d``/``-z``/``-v``/``-j``.

    ``-o PATH``, ``--output PATH``
        Output file path. The extension controls the output format: ``.szplt``,
        ``.plt``, or ``.dat``. Defaults to ``<stem>_aux<ext>``, or
        ``<stem>_no_aux<ext>`` with ``--strip``. Not used by ``--export-json`` alone
        which always writes ``<stem>_aux.json``, regardless of ``-o``.

    ``-f``, ``--force``
        Overwrite the output file(s) if they already exist. Without this flag the
        command exits with an error rather than silently clobbering an existing file.

:JSON Format:
    .. code:: json

        {
          "AUXDATASET": {"Solver": "MyCFD", "Version": "2.1"},
          "AUXZONE": {
            "0": {"Description": "Wing"},
            "all": {"Batch": "2024"}
          },
          "AUXVAR": {
            "Pressure": {"Units": "Pa"},
            "0": {"Source": "Experiment"}
          }
        }

    All three top-level keys are optional, and match this library's own attribute/method
    names for the same three levels exactly
    (``Write.auxdataset``/``add_auxdataset_dict``, ``Write.auxvar``/``add_auxvar_dict``;
    ``AUXZONE`` is the natural third member of that family even though zone-level aux
    has no "zone"-prefixed name internally.

    In ``"AUXZONE"``/``"AUXVAR"``, a key is either a 0-based index, a variable name
    (``"AUXVAR"`` only), or the literal string ``"all"`` meaning every zone/variable
    (the same three forms ``-z``/``-v`` accept on the command line). Every JSON key must
    be a quoted string, including numeric indices (``"0"``, not ``0``).

    ``--export-json`` writes this exact format back out, keyed by exact 0-based index
    (never ``"all"``, even if every zone happens to share identical aux content) and
    omitting any zone/variable/level with nothing to report.

:Returns:
    In the default mode, a new Tecplot file written to the output path with the
    requested auxiliary data merged in. With ``--strip``, a new Tecplot file with all
    auxiliary data removed. With ``--export-json``, a JSON file of everything that was
    found (and the source file is untouched, unless ``--strip`` is also given). Exit
    code is ``0`` on success and non-zero if the input file cannot be read, a
    ``-z``/``-v``/JSON target cannot be resolved, ``--strip``/``--export-json`` is
    combined with ``-d``/``-z``/``-v``/``-j``, or an output file already exists and
    ``--force`` is not set.

Examples:
    Tag a dataset with solver metadata (repeat the flag for multiple pairs)::

        $ tecaux -d Solver=MyCFD -d Version=2.1 flow.szplt

    Two pairs on zone 0, one pair on zone 1::

        $ tecaux -z 0 Description=Wing -z 0 Area=120sqm -z 1 Description=Fuselage \
              flow.szplt

    Annotate a single variable by name::

        $ tecaux -v Pressure Units=Pa flow.szplt

    Every zone at once, via the "all" target::

        $ tecaux -z all RunDate=2024-01-15 flow.plt

    Everything in one pass, written as a bash script would naturally lay it out::

        $ tecaux --data Solver=MyCFD \
              --data Version=2.1 \
              --zone 0 Case=A \
              --zone 0 Description=Wing \
              --zone 1 Case=B \
              --variable Pressure Units=Pa \
              -o tagged.szplt flow.szplt

    Bulk metadata from a file, with one CLI override on top::

        $ tecaux -j metadata.json -d Version=2.2 flow.szplt

    Back up everything currently on a file before editing it by hand, without touching
    the file itself::

        $ tecaux --export-json flow.szplt        # writes flow_aux.json

    Sanitize a file for external sharing, discarding whatever aux data it had::

        $ tecaux --strip flow.szplt               # writes flow_no_aux.szplt

    Sanitize a file but keep a record of what was removed, in case it's needed later
    (e.g. to reapply with ``-j`` after review)::

        $ tecaux --strip --export-json flow.szplt
        # writes flow_no_aux.szplt and flow_aux.json

    Call directly from a Python session::

        import tecio.cli.tecaux.main as tecaux

        tecaux(["-d", "Solver=MyCFD", "flow.szplt"])

See Also:
    * :mod:`tecio.cli.tecfix` - Rewrite a file with invalid variable arrays set to
      passive, using the same verbatim zone-copy approach.
    * :mod:`tecio.cli.teconvert` - Convert between formats without modifying auxiliary
      data.

"""

from __future__ import annotations

import argparse
import json
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


class _ArgError(Exception):
    """Raised for a malformed KEY=VALUE pair or unresolvable target."""


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecaux",
        description=(
            # -|-------------------|---------------------------------------------|
            "Add dataset-, zone-, or variable-level auxiliary data to a Tecplot\n"
            "file in a single read/write pass. Everything else (zones, variables,\n"
            "and sharing) is copied verbatim."
        ),
        epilog=(
            # -|-------------------|---------------------------------------------|
            "Example usage:\n"
            "  Dataset-level metadata (repeat the flag for multiple pairs)\n"
            "    $ tecaux -d Solver=MyCFD -d Version=2.1 <file>\n"
            "  Two pairs on zone 0, one pair on zone 1\n"
            "    $ tecaux -z 0 Description=Wing -z 0 Area=120sqm -z 1 Case=B <file>\n"
            "  A single variable by name\n"
            "    $ tecaux -v Pressure Units=Pa <file>\n"
            "  Every zone at once\n"
            "    $ tecaux -z all RunDate=2024-01-15 <file>\n"
            "  Bulk metadata from a file\n"
            "    $ tecaux -j metadata.json <file>\n"
            "  Export existing aux data without touching the source file\n"
            "    $ tecaux --export-json <file>\n"
            "  Strip all aux data, keeping a JSON backup of what was removed\n"
            "    $ tecaux --strip --export-json <file>\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file to add auxiliary data to.",
    )
    parser.add_argument(
        "-d",
        "--data",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="A name=value pair to set as dataset-level auxiliary data. Repeatable.",
    )
    parser.add_argument(
        "-z",
        "--zone",
        action="append",
        nargs=2,
        default=None,
        metavar=("INDEX", "KEY=VALUE"),
        help=(
            "A name=value pair to set as zone-level auxiliary data on the "
            "given 0-based zone index, or on every zone if INDEX is "
            "'all'. Repeatable, one pair per occurrence."
        ),
    )
    parser.add_argument(
        "-v",
        "--variable",
        dest="var",
        action="append",
        nargs=2,
        default=None,
        metavar=("INDEX_OR_NAME", "KEY=VALUE"),
        help=(
            "A name=value pair to set as variable-level auxiliary data on "
            "the given variable (0-based index or name), or on every "
            "variable if the target is 'all'. Repeatable, one pair per "
            "occurrence."
        ),
    )
    parser.add_argument(
        "-j",
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Load bulk auxiliary data from a JSON file, applied before "
            "-d/-z/-v (which override it on a key collision). Top-level "
            "keys: AUXDATASET, AUXZONE, AUXVAR (matching Write.auxdataset/"
            "Write.auxvar). Full format in the module docstring."
        ),
    )
    parser.add_argument(
        "-s",
        "--strip",
        action="store_true",
        default=False,
        help=(
            "Remove all auxiliary data (dataset, zone, and variable levels) "
            "and write the result to <stem>_no_aux<ext> (or -o's path). "
            "Mutually exclusive with -d/-z/-v/-j; combine with "
            "--export-json to also save what was removed."
        ),
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        default=False,
        help=(
            "Write every existing auxiliary entry to <stem>_aux.json, in "
            "the same AUXDATASET/AUXZONE/AUXVAR format -j reads -- without "
            "modifying the source file, unless combined with --strip. "
            "Mutually exclusive with -d/-z/-v/-j."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path. The extension controls the output format. "
            "Defaults to <stem>_aux<ext> in the same directory as the input "
            "(or <stem>_no_aux<ext> with --strip; not used by --export-json "
            "alone, which always writes <stem>_aux.json)."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_kv(token: str) -> tuple[str, str]:
    """Parse one ``KEY=VALUE`` token.

    Args:
        token: A single string like ``"Solver=MyCFD"``. Only the first ``"="`` is
               significant, so a value may itself contain ``"="``
               (e.g. ``"Formula=a=b+c"``). A value containing spaces must be quoted as
               a whole on the command line, the same as any other shell argument
               (e.g. ``-z 1 "Description=Wing surface"``) -- this function only ever
               sees whatever single token the shell already produced, so it can't
               recover a value that arrived pre-split into several tokens.

    Returns:
        ``(key, value)``, with the key stripped of surrounding whitespace.

    Raises:
        _ArgError: If *token* has no ``"="``, or the key is empty.

    """
    if "=" not in token:
        raise _ArgError(f"Expected KEY=VALUE, got: {token!r}")
    key, _, value = token.partition("=")
    key = key.strip()
    if not key:
        raise _ArgError(f"Empty key in: {token!r}")
    return key, value


def _resolve_variable(spec: str, var_names: list[str]) -> int | None:
    """Return a 0-based variable index from a name or 0-based integer string.

    Args:
        spec:      User-supplied string (e.g. ``"2"`` or ``"Pressure"``).
        var_names: Ordered list of variable names from the reader.

    Returns:
        0-based variable index, or ``None`` if *spec* cannot be resolved. Callers are
        responsible for reporting the failure.

    """
    try:
        idx = int(spec)
    except ValueError:
        pass
    else:
        if idx < 0 or idx >= len(var_names):
            return None
        return idx

    spec_lower = spec.lower()
    for i, name in enumerate(var_names):
        if name.lower() == spec_lower:
            return i

    return None


def _to_groups(
    raw_pairs: list[list[str]] | None,
) -> list[tuple[str | None, dict[str, str]]]:
    """Convert argparse's ``[[target, "KEY=VALUE"], ...]`` into consolidation input.

    Each occurrence becomes its own single-pair group; ``_consolidate_groups`` merges
    repeats of the same target on its own, so nothing needs to be pre-grouped here.

    Args:
        raw_pairs: ``args.zone``/``args.var`` as argparse produced them, or ``None`` if
                   the flag was never given.

    Returns:
        ``[(target, {key: value}), ...]``, target ``None`` for ``"all"``.

    Raises:
        _ArgError: If a KEY=VALUE token is malformed.

    """
    groups: list[tuple[str | None, dict[str, str]]] = []
    for target_raw, kv_raw in raw_pairs or []:
        target = None if target_raw.lower() == "all" else target_raw
        key, value = _parse_kv(kv_raw)
        groups.append((target, {key: value}))
    return groups


def _consolidate_groups(
    groups: list[tuple[str | None, dict[str, str]]],
    resolve: Any,
) -> tuple[dict[str, str], dict[int, dict[str, str]]]:
    """Merge repeated groups into one broadcast dict and one per-target dict.

    Args:
        groups:  ``[(target, {key: value}), ...]`` -- target is ``None`` ("every
                 zone"/"every variable") or a raw string to resolve.
        resolve: Callable taking the raw target string and returning a 0-based index for
                 a specific target, or raising ``_ArgError`` if it can't be resolved.

    Returns:
        ``(broadcast, by_target)`` where ``by_target`` is keyed by whatever index
        ``resolve`` returns.

    """
    broadcast: dict[str, str] = {}
    by_target: dict[int, dict[str, str]] = {}
    for target, pairs in groups:
        if target is None:
            broadcast.update(pairs)
        else:
            idx = resolve(target)
            by_target.setdefault(idx, {}).update(pairs)
    return broadcast, by_target


# --------------------------------------------------------------------------------------
# JSON loading
# --------------------------------------------------------------------------------------


def _load_json_aux(
    path: str,
) -> tuple[
    dict[str, str],
    list[tuple[str | None, dict[str, str]]],
    list[tuple[str | None, dict[str, str]]],
]:
    """Load bulk auxiliary data from a JSON file.

    Top-level keys are ``AUXDATASET``, ``AUXZONE``, ``AUXVAR``.

    Returns:
        ``(dataset_aux, zone_groups, var_groups)`` in the same shape ``_to_groups``
        produces, so both sources merge identically.

    Raises:
        _ArgError: If the file can't be read or parsed, contains an unrecognized
                   top-level key, or a value isn't a string -> string mapping.

    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise _ArgError(f"Could not read JSON file {path!r}: {exc}") from exc

    if not isinstance(data, dict):
        raise _ArgError(f"JSON file {path!r} must contain an object at the top level.")

    _known_keys = {"AUXDATASET", "AUXZONE", "AUXVAR"}
    unknown: set[str] = {str(k) for k in data} - _known_keys
    if unknown:
        raise _ArgError(
            f"Unrecognized top-level key(s) in {path!r}: {sorted(unknown)}.  "
            f"Expected one of: {sorted(_known_keys)}."
        )

    def _as_str_dict(obj: Any, where: str) -> dict[str, str]:
        if not isinstance(obj, dict):
            raise _ArgError(f"{where} must be an object of name: value strings.")
        return {str(k): str(v) for k, v in obj.items()}

    dataset_aux = _as_str_dict(data.get("AUXDATASET", {}), "'AUXDATASET'")

    zone_groups: list[tuple[str | None, dict[str, str]]] = []
    for key, val in data.get("AUXZONE", {}).items():
        target = None if str(key).lower() == "all" else str(key)
        zone_groups.append((target, _as_str_dict(val, f"'AUXZONE.{key}'")))

    var_groups: list[tuple[str | None, dict[str, str]]] = []
    for key, val in data.get("AUXVAR", {}).items():
        target = None if str(key).lower() == "all" else str(key)
        var_groups.append((target, _as_str_dict(val, f"'AUXVAR.{key}'")))

    return dataset_aux, zone_groups, var_groups


def _collect_all_aux(reader: TecplotReader) -> dict[str, Any]:
    """Collect every dataset-, zone-, and variable-level aux entry from *reader*.

    The exact inverse of ``_load_json_aux``: produces the same
    ``AUXDATASET``/``AUXZONE``/``AUXVAR`` structure, so a file exported with
    ``--export-json`` can be fed straight back in with ``-j`` (to the same
    file, a modified copy, or an entirely different one) without any
    reshaping. Zones/variables are keyed by their exact 0-based index --
    never collapsed into an ``"all"`` entry even when every zone happens to
    share the same aux content, since a later zone added to the file
    wouldn't have had that entry originally and shouldn't silently inherit
    it on a subsequent import.

    A zone or variable with no aux entries at all is omitted entirely
    (sparse), not written as an empty object -- matching how a missing key
    on the *input* side already means "nothing to apply here".

    Args:
        reader: An open ``Read`` instance.

    Returns:
        A dict with up to three top-level keys (``AUXDATASET``, ``AUXZONE``,
        ``AUXVAR``); a level with nothing to export is omitted entirely, so
        a file with no aux data anywhere yields ``{}``.

    """
    result: dict[str, Any] = {}

    dataset_aux = dict(reader.auxdata.items())
    if dataset_aux:
        result["AUXDATASET"] = dataset_aux

    zone_aux: dict[str, dict[str, str]] = {}
    for i, zone in enumerate(reader.zones):
        entries = dict(zone.auxdata.items())
        if entries:
            zone_aux[str(i)] = entries
    if zone_aux:
        result["AUXZONE"] = zone_aux

    var_aux: dict[str, dict[str, str]] = {}
    for i in range(reader.num_vars):
        entries = dict(reader.get_var_auxdata(i).items())
        if entries:
            var_aux[str(i)] = entries
    if var_aux:
        result["AUXVAR"] = var_aux

    return result


# --------------------------------------------------------------------------------------
# Per-zone processing
# --------------------------------------------------------------------------------------


def _process_zone(
    zone: TecplotZoneReader,
) -> tuple[list[np.ndarray], list[Any], set[str], dict[str, int], dict[str, str]]:
    """Copy one zone's variable data/metadata verbatim.

    Sharing references are passed through unchanged: this tool writes every zone in the
    same order as the source, so source zone N is always output zone N.

    Returns:
        A 5-tuple: ``(data, value_locations, passive_vars, var_sharing,
        existing_aux)``, where the first two cover active, non-shared variables only.
    """
    data: list[np.ndarray] = []
    value_locations: list[Any] = []
    passive_vars: set[str] = set()
    var_sharing: dict[str, int] = {}

    for var in zone.variables:
        sv = var.shared_zone  # 0-based source zone index, or None

        if var.is_passive():
            passive_vars.add(var.name)
            continue
        if sv is not None:
            var_sharing[var.name] = sv
            continue

        arr = var.values
        if arr is None or arr.size == 0:
            passive_vars.add(var.name)
        else:
            data.append(arr)
            value_locations.append(var.value_location)

    existing_aux: dict[str, str] = {}
    if len(zone.auxdata) > 0:
        existing_aux = dict(zone.auxdata.items())

    return data, value_locations, passive_vars, var_sharing, existing_aux


def _write_zone_data(
    writer: TecplotWriter,
    zone: TecplotZoneReader,
    writer_data: list[np.ndarray],
    writer_locs: list[Any],
    passive_vars: set[str],
    var_sharing: dict[str, int],
    zone_aux: dict[str, str] | None,
) -> None:
    """Perform the actual write zone call for one already processed zone.

    Shared between add-mode (``main()``) and strip/export-mode
    (``_run_strip_or_export()``) -- the only thing that differs between them
    is what *zone_aux* is (a merged dict, or ``None`` when stripping), so
    this is the one piece worth keeping in exactly one place rather than
    risking the ``con_sharing``/``node_map`` handling drifting between two
    copies.

    Args:
        writer:       Open writer instance for the destination file.
        zone:         Source zone reader being copied.
        writer_data:  Active, non-shared variable arrays (from ``_process_zone``).
        writer_locs:  Matching value locations (from ``_process_zone``).
        passive_vars: Variable names marked passive.
        var_sharing:  ``{variable name: source zone}`` mapping.
        zone_aux:     Zone-level aux to write, or ``None`` for none at all.

    """
    zt = zone.zone_type
    common_kw: dict[str, Any] = {
        "title": zone.title,
        "value_locations": writer_locs,
        "passive_vars": passive_vars,
        "var_sharing": var_sharing,
        "solution_time": zone.solution_time,
        "strand_id": zone.strand_id,
        "aux": zone_aux,
    }

    if isinstance(zone, TecplotOrderedZoneReader):
        writer.write_ordered_zone(data=writer_data, **common_kw)
    elif isinstance(zone, TecplotPolyZoneReader):
        con_sharing = zone.shared_connectivity
        writer.write_poly_zone(
            zone_type=zt,
            data=writer_data,
            con_sharing=con_sharing,
            **({} if con_sharing is not None else poly_face_kwargs(zone)),
            **common_kw,
        )
    elif isinstance(zone, TecplotFEZoneReader):
        con_sharing = zone.shared_connectivity
        fe_kw = common_kw.copy()

        # Face-neighbor connections
        if zone.face_neighbor_mode is not None and not isinstance(
            writer, TecplotSzlWriter
        ):
            fe_kw["face_neighbors"] = zone.get_face_connections()
            fe_kw["face_neighbor_mode"] = zone.face_neighbor_mode
        writer.write_fe_zone(
            zone_type=zt,
            data=writer_data,
            node_map=None if con_sharing is not None else zone.node_map,
            con_sharing=con_sharing,
            **fe_kw,
        )
    else:
        raise NotImplementedError(
            f"Zone '{zone.title}' is neither an ordered, classic FE nor poly "
            "zone; writing is not supported for it."
        )


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def _run_strip_or_export(args: argparse.Namespace, src: Path) -> int:
    """Handle ``--strip``/``--export-json``, independently or together.

    Args:
        args: Parsed arguments; ``args.strip`` and/or ``args.export_json``
              is ``True`` (``main()`` only calls this when at least one is).
        src:  Validated-to-exist input file path.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    if args.output is not None and args.export_json and not args.strip:
        print(
            "Warning: -o is ignored by --export-json alone -- the export "
            f"always goes to {src.stem}_aux.json.",
            file=sys.stderr,
        )

    json_dst = src.with_name(f"{src.stem}_aux.json")
    strip_dst = (
        Path(args.output)
        if args.output is not None
        else src.with_stem(src.stem + "_no_aux")
    )

    if args.export_json and json_dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {json_dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1
    if args.strip and strip_dst.exists() and not args.force:
        print(
            (
                f"Error: output file already exists: {strip_dst}\n"
                "Use --force to overwrite."
            ),
            file=sys.stderr,
        )
        return 1

    try:
        with tecio_open(str(src), "r") as reader:
            if args.export_json:
                aux = _collect_all_aux(reader)
                with open(json_dst, "w", encoding="utf-8") as fh:
                    json.dump(aux, fh, indent=2)
                print(f"Exported auxiliary data: {src}  ->  {json_dst}")

            if args.strip:
                print(f"Stripping auxiliary data: {src}  ->  {strip_dst}")
                with tecio_open(
                    str(strip_dst),
                    "w",
                    title=reader.title,
                    variables=reader.variables,
                    file_type=reader.file_type,
                ) as writer:
                    for zone in reader.zones:
                        writer_data, writer_locs, passive_vars, var_sharing, _ = (
                            _process_zone(zone)
                        )
                        # No dataset/var aux is ever added to the writer, and zone_aux
                        # is unconditionally None here.
                        _write_zone_data(
                            writer,
                            zone,
                            writer_data,
                            writer_locs,
                            passive_vars,
                            var_sharing,
                            None,
                        )
                print(f"Done. Stripped output written to: {strip_dst}")

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        if args.strip:
            strip_dst.unlink(missing_ok=True)
        if args.export_json:
            json_dst.unlink(missing_ok=True)
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Add, remove, or export auxiliary data to a Tecplot file.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    if args.strip or args.export_json:
        add_flags_given = bool(args.data or args.zone or args.var or args.json)
        if add_flags_given:
            print(
                "Error: --strip/--export-json cannot be combined with "
                "-d/-z/-v/-j -- run them as separate calls.",
                file=sys.stderr,
            )
            return 1
        return _run_strip_or_export(args, src)

    dst = (
        Path(args.output)
        if args.output is not None
        else src.with_stem(src.stem + "_aux")
    )

    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        # JSON is the baseline; -d/-z/-v groups are appended after it, so they naturally
        # win on a key collision.
        dataset_aux: dict[str, str] = {}
        zone_groups: list[tuple[str | None, dict[str, str]]] = []
        var_groups: list[tuple[str | None, dict[str, str]]] = []

        if args.json is not None:
            j_dataset, j_zones, j_vars = _load_json_aux(args.json)
            dataset_aux.update(j_dataset)
            zone_groups.extend(j_zones)
            var_groups.extend(j_vars)

        for kv in args.data or []:
            k, v = _parse_kv(kv)
            dataset_aux[k] = v
        zone_groups.extend(_to_groups(args.zone))
        var_groups.extend(_to_groups(args.var))
    except _ArgError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not (dataset_aux or zone_groups or var_groups):
        print(
            "Warning: no -d, -z, -v, or -j given -- output will be a verbatim copy.",
            file=sys.stderr,
        )

    try:
        with tecio_open(str(src), "r") as reader:
            var_names: list[str] = reader.variables
            num_vars: int = reader.num_vars
            num_zones: int = reader.num_zones

            def _resolve_zone_target(raw: str) -> int:
                try:
                    idx = int(raw)
                except ValueError as exc:
                    raise _ArgError(
                        f"Zone target must be an index or 'all', got: {raw!r}"
                    ) from exc
                if idx < 0 or idx >= num_zones:
                    raise _ArgError(
                        f"Zone index {idx} out of range [0, {num_zones - 1}]."
                    )
                return idx

            def _resolve_var_target(raw: str) -> int:
                idx0 = _resolve_variable(raw, var_names)
                if idx0 is None:
                    raise _ArgError(
                        f"Could not resolve variable target {raw!r}.  "
                        f"Available: {var_names}"
                    )
                return idx0  # keyed 0-based, matching zone targets

            try:
                zone_broadcast, zone_by_target = _consolidate_groups(
                    zone_groups, _resolve_zone_target
                )
                var_broadcast, var_by_target = _consolidate_groups(
                    var_groups, _resolve_var_target
                )
            except _ArgError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1

            print(f"Adding auxiliary data: {src}  ->  {dst}")
            if dataset_aux:
                print(f"  Dataset : {dataset_aux}")
            for target, pairs in zone_groups:
                where = f"zone {target}" if target is not None else "every zone"
                print(f"  Zone    : {pairs}  ({where})")
            for target, pairs in var_groups:
                where = f"variable {target}" if target is not None else "every variable"
                print(f"  Variable: {pairs}  ({where})")

            with tecio_open(
                str(dst),
                "w",
                title=reader.title,
                variables=var_names,
                file_type=reader.file_type,
            ) as writer:
                merged_dataset_aux = {**dict(reader.auxdata.items()), **dataset_aux}
                if merged_dataset_aux:
                    writer.add_auxdataset_dict(merged_dataset_aux)

                auxvar: dict[int, dict[str, str]] = {}
                for i in range(num_vars):
                    existing = dict(reader.get_var_auxdata(i).items())
                    merged = {
                        **existing,
                        **var_broadcast,
                        **var_by_target.get(i, {}),
                    }
                    if merged:
                        auxvar[i] = merged
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                # add_auxdataset_dict()/add_auxvar_dict() only buffer; the write happens
                # in flush_aux(), normally auto-triggered by the *lazy*-open path on the
                # first zone write. Passing variables= above means this writer is
                # already open (eager), so that automatic trigger never fires.
                writer.flush_aux()

                for zone_num, zone in enumerate(reader.zones):
                    (
                        writer_data,
                        writer_locs,
                        passive_vars,
                        var_sharing,
                        existing_zone_aux,
                    ) = _process_zone(zone)

                    merged_zone_aux = {
                        **existing_zone_aux,
                        **zone_broadcast,
                        **zone_by_target.get(zone_num, {}),
                    } or None

                    _write_zone_data(
                        writer,
                        zone,
                        writer_data,
                        writer_locs,
                        passive_vars,
                        var_sharing,
                        merged_zone_aux,
                    )

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
