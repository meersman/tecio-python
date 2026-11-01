#!/usr/bin/env python3
"""pytest tests for :mod:`tecio._io` — the ``tecio.open()`` entry point.

Covers:
* All five open modes: ``'r'``, ``'w'``, ``'x'``, ``'a'``, ``'a+'``.
* :class:`~tecio._io.AppendWrite` — append new zones to an existing file.
* :class:`~tecio._io.AppendReadWrite` — append while reading the original
  data in the same session.
* Error paths (wrong extension, ``'x'`` on existing file, ``'a'`` on
  missing file, FEPOLYGON in append source).
* Cross-format round-trip via append (read SZL, write PLT, verify counts).

These tests are format-agnostic at the ``tecio.open()`` level. All writes
use ``.szplt`` unless the test specifically targets PLT/DAT behaviour.
"""

# ruff: noqa: E501

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import tecio
from tecio import FileType, ZoneType
from tecio._io import AppendReadWrite, AppendWrite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(tecio.__file__).parent.parent / "tests"
_ONERA = {fmt: _TESTS_DIR / f"Onera.{fmt}" for fmt in ("szplt", "plt", "dat")}

_RTOL_F32 = 1e-5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_simple_szplt(path: Path, n_zones: int = 1) -> dict:
    """Write a trivial SZL file with *n_zones* identical IJK zones.

    Returns:
        dict: ndarrays for keys ``x`` and ``c`` after writing to file.
    """
    x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    c = np.sin(2 * np.pi * x).astype(np.float32)

    with tecio.open(str(path), "w", variables=["x", "c"], title="test") as w:
        for i in range(n_zones):
            w.write_ordered_zone(
                data=[x, c],
                title=f"zone_{i + 1}",
                solution_time=float(i),
                strand_id=1 if n_zones > 1 else 0,
            )
    return {"x": x, "c": c}


def _write_simple_plt(path: Path, n_zones: int = 1) -> dict:
    """Write a trivial PLT file with *n_zones* identical IJK zones.

    Returns:
        dict: ndarrays for keys ``x`` and ``c`` after writing to file.
    """
    x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    c = np.cos(2 * np.pi * x).astype(np.float32)

    with tecio.open(str(path), "w", variables=["x", "c"], title="test_plt") as w:
        for i in range(n_zones):
            w.write_ordered_zone(
                data=[x, c],
                title=f"zone_{i + 1}",
                solution_time=float(i),
                strand_id=1 if n_zones > 1 else 0,
            )
    return {"x": x, "c": c}


def _write_simple_dat(path: Path, n_zones: int = 1) -> dict:
    """Write a trivial DAT file with *n_zones* identical IJK zones.

    Returns:
        dict: ndarrays for keys ``x`` and ``c`` after writing to file.
    """
    x = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    c = (x**2).astype(np.float32)

    with tecio.open(str(path), "w", variables=["x", "c"], title="test_dat") as w:
        for i in range(n_zones):
            w.write_ordered_zone(data=[x, c], title=f"zone_{i + 1}")
    return {"x": x, "c": c}


# ===========================================================================
# Mode 'r' — read
# ===========================================================================


class TestOpenRead:
    """Tests for ``tecio.open(path, 'r')``."""

    @pytest.mark.parametrize("fmt", ["szplt", "plt", "dat"])
    def test_open_onera(self, fmt: str) -> None:
        """Onera files open without error in all three formats."""
        with tecio.open(str(_ONERA[fmt]), "r") as r:
            assert r.num_vars == 18
            assert r.num_zones == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Opening a non-existent file raises :exc:`FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            tecio.open(str(tmp_path / "ghost.szplt"), "r")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """An unrecognised extension raises :exc:`ValueError`."""
        path = tmp_path / "data.nc"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            tecio.open(str(path), "r")

    def test_default_mode_is_read(self, tmp_path: Path) -> None:
        """Omitting ``mode`` defaults to ``'r'``."""
        path = tmp_path / "simple.szplt"
        _write_simple_szplt(path)
        with tecio.open(str(path)) as r:
            assert r.num_vars == 2

    def test_context_manager_read(self, tmp_path: Path) -> None:
        """Reader works as a context manager (no error on ``__exit__``)."""
        path = tmp_path / "ctx.szplt"
        _write_simple_szplt(path)
        with tecio.open(str(path), "r") as r:
            zones = r.num_zones
        assert zones == 1


# ===========================================================================
# Mode 'w' — write (overwrite)
# ===========================================================================


class TestOpenWrite:
    """Tests for ``tecio.open(path, 'w')``."""

    def test_creates_new_file(self, tmp_path: Path) -> None:
        """Mode ``'w'`` creates a file that did not previously exist."""
        path = tmp_path / "new.szplt"
        assert not path.exists()
        with tecio.open(str(path), "w") as w:
            x = np.array([0.0, 1.0], dtype=np.float32)
            w.write_ordered_zone(data=[x], variables=["x"])
        assert path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Mode ``'w'`` silently replaces an existing file."""
        path = tmp_path / "overwrite.szplt"
        _write_simple_szplt(path, n_zones=3)

        with tecio.open(str(path), "w") as w:
            x = np.array([0.0, 1.0], dtype=np.float32)
            w.write_ordered_zone(data=[x], variables=["x"])

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 1  # not 3

    @pytest.mark.parametrize(
        "fmt,ext", [("szplt", ".szplt"), ("plt", ".plt"), ("dat", ".dat")]
    )
    def test_write_all_formats(self, fmt: str, ext: str, tmp_path: Path) -> None:
        """Mode ``'w'`` works for all supported extensions."""
        path = tmp_path / f"out{ext}"
        writer_fn = {
            "szplt": _write_simple_szplt,
            "plt": _write_simple_plt,
            "dat": _write_simple_dat,
        }[fmt]
        writer_fn(path)
        with tecio.open(str(path), "r") as r:
            assert r.num_vars == 2


# ===========================================================================
# Mode 'x' — exclusive creation
# ===========================================================================


class TestOpenExclusive:
    """Tests for ``tecio.open(path, 'x')``."""

    def test_creates_new_file(self, tmp_path: Path) -> None:
        """Mode ``'x'`` creates a file when it does not exist."""
        path = tmp_path / "exclusive.szplt"
        with tecio.open(str(path), "x") as w:
            x = np.array([0.0, 1.0], dtype=np.float32)
            w.write_ordered_zone(data=[x], variables=["x"])
        assert path.exists()

    def test_raises_if_file_exists(self, tmp_path: Path) -> None:
        """Mode ``'x'`` raises :exc:`FileExistsError` on an existing file."""
        path = tmp_path / "exists.szplt"
        _write_simple_szplt(path)
        with pytest.raises(FileExistsError):
            tecio.open(str(path), "x")

    def test_exclusive_content_is_correct(self, tmp_path: Path) -> None:
        """Data written via ``'x'`` reads back correctly."""
        path = tmp_path / "xwrite.szplt"
        x_in = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        with tecio.open(str(path), "x") as w:
            w.write_ordered_zone(data=[x_in], variables=["x"])

        with tecio.open(str(path), "r") as r:
            np.testing.assert_allclose(
                r.zones[0].variables[0].values.ravel(), x_in, rtol=_RTOL_F32
            )


# ===========================================================================
# Mode 'a' — append (AppendWrite)
# ===========================================================================


class TestAppendWrite:
    """Tests for ``tecio.open(path, 'a')`` returning :class:`AppendWrite`."""

    def test_returns_append_write_instance(self, tmp_path: Path) -> None:
        """``open(..., 'a')`` returns an :class:`AppendWrite`."""
        path = tmp_path / "aw.szplt"
        _write_simple_szplt(path)
        with tecio.open(str(path), "a") as w:
            assert isinstance(w, AppendWrite)

    def test_zone_count_after_append_szplt(self, tmp_path: Path) -> None:
        """Appending one zone to a 1-zone SZL file produces 2 zones."""
        path = tmp_path / "append_szplt.szplt"
        arrays = _write_simple_szplt(path)

        with tecio.open(str(path), "a") as w:
            assert w.current_zone == 0  # one zone was copied from original
            w.write_ordered_zone(
                data=[arrays["x"], arrays["c"]],
                title="appended_zone",
                solution_time=99.0,
                strand_id=1,
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            assert r.zones[1].title == "appended_zone"
            assert r.zones[1].solution_time == pytest.approx(99.0)

    def test_zone_count_after_append_plt(self, tmp_path: Path) -> None:
        """Appending one zone to a 2-zone PLT file produces 3 zones."""
        path = tmp_path / "append_plt.plt"
        arrays = _write_simple_plt(path, n_zones=2)

        with tecio.open(str(path), "a") as w:
            assert w.current_zone == 1  # two zones copied
            w.write_ordered_zone(
                data=[arrays["x"], arrays["c"]],
                title="new_zone",
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 3

    def test_original_zones_preserved(self, tmp_path: Path) -> None:
        """Zones from the original file survive the append operation intact."""
        path = tmp_path / "preserve.szplt"
        orig = _write_simple_szplt(path)

        with tecio.open(str(path), "a") as w:
            extra = np.ones(10, dtype=np.float32) * 9.9
            w.write_ordered_zone(data=[extra, extra], title="extra")

        with tecio.open(str(path), "r") as r:
            # Original zone 1 values must be unchanged.
            np.testing.assert_allclose(
                r.zones[0].variables[0].values.ravel(),
                orig["x"],
                rtol=_RTOL_F32,
            )

    def test_variable_list_preserved(self, tmp_path: Path) -> None:
        """Variable names from the original file are unchanged after append."""
        path = tmp_path / "varnames.szplt"
        _write_simple_szplt(path)

        with tecio.open(str(path), "a") as w:
            original_vars = w.variables
            x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
            w.write_ordered_zone(data=[x, x], title="z2")

        with tecio.open(str(path), "r") as r:
            assert r.variables == original_vars

    def test_title_preserved_by_default(self, tmp_path: Path) -> None:
        """Original dataset title is preserved when not overridden."""
        path = tmp_path / "title.szplt"
        with tecio.open(str(path), "w", title="MyTitle") as w:
            x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
            w.write_ordered_zone(data=[x], variables=["x"])

        with tecio.open(str(path), "a") as w:
            assert w.title == "MyTitle"

    def test_append_multiple_new_zones(self, tmp_path: Path) -> None:
        """Multiple zones can be appended in a single ``'a'`` session."""
        path = tmp_path / "multi_append.szplt"
        _write_simple_szplt(path, n_zones=1)
        x = np.linspace(0.0, 1.0, 10, dtype=np.float32)

        with tecio.open(str(path), "a") as w:
            for i in range(3):
                w.write_ordered_zone(data=[x, x], title=f"new_{i}")

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 4  # 1 original + 3 appended

    def test_append_fe_zone(self, tmp_path: Path) -> None:
        """An FE zone can be appended to a file containing an ordered zone."""
        path = tmp_path / "append_fe.szplt"
        _write_simple_szplt(path)

        pts = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.0, 1.0]], dtype=np.float32
        )
        nodes = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        x, c = pts[:, 0], np.zeros(4, dtype=np.float32)

        with tecio.open(str(path), "a") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, c],
                node_map=nodes,
                title="tri_zone",
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            assert r.zones[1].zone_type == ZoneType.FETRIANGLE

    def test_append_to_missing_file_raises(self, tmp_path: Path) -> None:
        """Appending to a non-existent file raises :exc:`FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            tecio.open(str(tmp_path / "ghost.szplt"), "a")

    def test_close_replaces_original(self, tmp_path: Path) -> None:
        """The temporary file replaces the original on close (not before)."""
        path = tmp_path / "replace.szplt"
        _write_simple_szplt(path)
        mtime_before = path.stat().st_mtime

        with tecio.open(str(path), "a") as w:
            x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
            w.write_ordered_zone(data=[x, x], title="z2")

        # mtime must have changed because the file was atomically replaced.
        assert path.stat().st_mtime != mtime_before

    def test_onera_append_increases_zones(self, tmp_path: Path) -> None:
        """Appending one zone to the Onera SZL file produces 3 zones."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)

        with tecio.open(str(src), "r") as r:
            orig_vars = r.variables
            # r.zones[0].num_nodes

        np.zeros(10, dtype=np.float32)
        with tecio.open(str(src), "a") as w:
            assert w.current_zone == 1  # two Onera zones copied
            # Append a new tiny zone (all non-Onera variables passive).
            n_vars = len(w.variables)
            passive_vars = set(range(3, n_vars))  # everything except x, y, z
            pts = np.zeros((10, 3), dtype=np.float32)
            nodes = np.array([[i, i + 1] for i in range(9)], dtype=np.int32)
            w.write_fe_zone(
                zone_type=ZoneType.FELINESEG,
                data=[pts[:, 0], pts[:, 1], pts[:, 2]],
                node_map=nodes,
                passive_vars=passive_vars,
                title="tiny_new",
            )

        with tecio.open(str(src), "r") as r:
            assert r.num_zones == 3
            assert r.variables == orig_vars

    def test_append_preserves_zone_aux_data(self, tmp_path: Path) -> None:
        """Zone-level aux data from the original file survives append."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)

        with tecio.open(str(src), "a") as w:
            x = np.zeros(5, dtype=np.float32)
            n_vars = len(w.variables)
            passive_vars = set(range(3, n_vars))  # everything except x, y, z
            nodes = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
            w.write_fe_zone(
                zone_type=ZoneType.FELINESEG,
                data=[x, x, x],
                node_map=nodes,
                passive_vars=passive_vars,
                title="filler",
            )

        with tecio.open(str(src), "r") as r:
            # Zone 2 (WingSurface) has BoundaryCondition aux data.
            aux = r.zones[1].auxdata
            assert aux["Common.BoundaryCondition"] == "Wall"


# ===========================================================================
# Mode 'a+' — append-read (AppendReadWrite)
# ===========================================================================


class TestAppendReadWrite:
    """Tests for ``tecio.open(path, 'a+')`` returning :class:`AppendReadWrite`."""

    def test_returns_append_read_write_instance(self, tmp_path: Path) -> None:
        """``open(..., 'a+')`` returns an :class:`AppendReadWrite`."""
        path = tmp_path / "arw.szplt"
        _write_simple_szplt(path)
        with tecio.open(str(path), "a+") as rw:
            assert isinstance(rw, AppendReadWrite)

    def test_can_read_original_zones(self, tmp_path: Path) -> None:
        """Original zones are readable through the read interface."""
        path = tmp_path / "arw_read.szplt"
        orig = _write_simple_szplt(path)

        with tecio.open(str(path), "a+") as rw:
            assert rw.num_zones == 1
            np.testing.assert_allclose(
                rw.zones[0].variables[0].values.ravel(),
                orig["x"],
                rtol=_RTOL_F32,
            )

    def test_can_write_new_zone_while_reading(self, tmp_path: Path) -> None:
        """New zones can be written while reading the original."""
        path = tmp_path / "arw_write.szplt"
        orig = _write_simple_szplt(path)

        with tecio.open(str(path), "a+") as rw:
            # Read original data.
            x_orig = rw.zones[0].variables[0].values.ravel().copy()
            # Write a derived zone.
            x_new = (x_orig * 2.0).astype(np.float32)
            c_new = orig["c"] * 0.5
            rw.write_ordered_zone(data=[x_new, c_new], title="derived")

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            np.testing.assert_allclose(
                r.zones[1].variables[0].values.ravel(), x_new, rtol=_RTOL_F32
            )

    def test_read_interface_exposes_original_metadata(self, tmp_path: Path) -> None:
        """``num_vars``, ``variables``, ``title``, ``file_type`` match the source."""
        path = tmp_path / "arw_meta.szplt"
        with tecio.open(str(path), "w", title="MyTitle", file_type=FileType.FULL) as w:
            x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
            w.write_ordered_zone(data=[x, x], variables=["x", "c"])

        with tecio.open(str(path), "a+") as rw:
            assert rw.title == "MyTitle"
            assert rw.file_type == FileType.FULL
            assert rw.num_vars == 2
            assert rw.variables == ["x", "c"]

    def test_num_zones_reports_original_count(self, tmp_path: Path) -> None:
        """``num_zones`` reports the count before any new zones are appended."""
        path = tmp_path / "arw_nz.szplt"
        _write_simple_szplt(path, n_zones=3)

        with tecio.open(str(path), "a+") as rw:
            assert rw.num_zones == 3  # original count — new writes not included
            x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
            rw.write_ordered_zone(data=[x, x], title="extra")
            assert rw.num_zones == 3  # still original count during session

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 4  # now includes the new zone

    def test_compute_and_write_average(self, tmp_path: Path) -> None:
        """A time-average computed from original zones is appended correctly."""
        path = tmp_path / "avg.szplt"
        n = 10
        times = np.linspace(0.0, np.pi, 5)
        x_base = np.linspace(0.0, 1.0, n, dtype=np.float32)

        with tecio.open(str(path), "w", variables=["x", "c"]) as w:
            for i, t in enumerate(times):
                c = np.sin(x_base + t).astype(np.float32)
                data = [x_base, c] if i == 0 else [c]
                sharing = None if i == 0 else {"x": 0}
                w.write_ordered_zone(
                    data=data,
                    var_sharing=sharing,
                    solution_time=t,
                    strand_id=1,
                )

        with tecio.open(str(path), "a+") as rw:
            assert rw.num_zones == len(times)
            c_sum = sum(
                rw.zones[i].variables[1].values.ravel().astype(np.float64)
                for i in range(rw.num_zones)
            )
            c_avg = (c_sum / rw.num_zones).astype(np.float32)
            rw.zones[0].variables[0].values.ravel()

            rw.write_ordered_zone(
                data=[c_avg],
                var_sharing={"x": 0},
                title="time_average",
                solution_time=float(rw.num_zones),
                strand_id=2,
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == len(times) + 1
            avg_zone = r.zones[-1]
            assert avg_zone.title == "time_average"
            assert avg_zone.strand_id == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """``'a+'`` on a non-existent file raises :exc:`FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            tecio.open(str(tmp_path / "ghost.szplt"), "a+")

    def test_onera_read_and_append(self, tmp_path: Path) -> None:
        """Onera file: read zone titles then append a small new zone."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)

        with tecio.open(str(src), "a+") as rw:
            assert rw.num_zones == 2
            titles = [rw.zones[i].title for i in range(rw.num_zones)]
            assert titles == ["FluidVolume", "WingSurface"]

            n_vars = len(rw.variables)
            passive_vars = set(range(3, n_vars))  # everything except x, y, z
            nodes = np.array([[0, 1], [1, 2]], dtype=np.int32)
            xyz = np.zeros(3, dtype=np.float32)
            rw.write_fe_zone(
                zone_type=ZoneType.FELINESEG,
                data=[xyz, xyz, xyz],
                node_map=nodes,
                passive_vars=passive_vars,
                title="dummy",
            )

        with tecio.open(str(src), "r") as r:
            assert r.num_zones == 3
            assert r.zones[2].title == "dummy"


# ===========================================================================
# Mode errors
# ===========================================================================


class TestOpenErrors:
    """Tests for invalid open() arguments."""

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        """An unrecognised mode string raises :exc:`ValueError`."""
        path = tmp_path / "test.szplt"
        _write_simple_szplt(path)
        with pytest.raises(ValueError, match="[Mm]ode"):
            tecio.open(str(path), "z")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """An unrecognised extension raises :exc:`ValueError`."""
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            tecio.open(str(tmp_path / "data.hdf5"), "w")


# ===========================================================================
# get_variable_list / get_zone_list
# ===========================================================================


class TestGetVariableAndZoneList:
    """Tests for ``tecio.get_variable_list``/``tecio.get_zone_list``."""

    @pytest.mark.parametrize("fmt", ["szplt", "plt", "dat"])
    def test_matches_full_open(self, fmt: str, tmp_path: Path) -> None:
        """Both functions agree with a full ``tecio.open()`` read, in every format."""
        path = tmp_path / f"simple.{fmt}"
        writer = {
            "szplt": _write_simple_szplt,
            "plt": _write_simple_plt,
            "dat": _write_simple_dat,
        }[fmt]
        writer(path, n_zones=3)

        assert tecio.get_variable_list(str(path)) == ["x", "c"]
        assert tecio.get_zone_list(str(path)) == ["zone_1", "zone_2", "zone_3"]

        with tecio.open(str(path), "r") as r:
            assert tecio.get_variable_list(str(path)) == list(r.variables)
            assert tecio.get_zone_list(str(path)) == [z.title for z in r.zones]

    def test_dat_unnamed_zone_returns_empty_string(self, tmp_path: Path) -> None:
        """A DAT zone with no ``T="..."`` contributes ``''``, not a placeholder."""
        path = tmp_path / "untitled.dat"
        path.write_text('TITLE="t"\nVARIABLES="x" "y"\nZONE I=2\n1.0 2.0\n3.0 4.0\n')
        assert tecio.get_zone_list(str(path)) == [""]

    def test_dat_multi_zone_no_ijk(self, tmp_path: Path) -> None:
        """The point-count-inference legacy format doesn't confuse zone scanning."""
        path = tmp_path / "no_ijk.dat"
        path.write_text(
            'TITLE="t"\nVARIABLES="x" "y"\n'
            'ZONE T="First", F=point\n1.0 10.0\n2.0 20.0\n'
            'ZONE T="Second", F=point\n100.0 1000.0\n200.0 2000.0\n'
        )
        assert tecio.get_zone_list(str(path)) == ["First", "Second"]
        assert tecio.get_variable_list(str(path)) == ["x", "y"]

    def test_dat_trailing_whitespace_line(self, tmp_path: Path) -> None:
        """A trailing whitespace-only line after the last zone doesn't break scanning."""
        path = tmp_path / "trailing.dat"
        path.write_text('Title="t"\nVariables="X" "Y"\nZONE F=point\n1 2\n3 4\n    \n')
        assert tecio.get_variable_list(str(path)) == ["X", "Y"]
        assert tecio.get_zone_list(str(path)) == [""]

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """An unrecognised extension raises :exc:`ValueError` for both functions."""
        path = tmp_path / "data.nc"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            tecio.get_variable_list(str(path))
        with pytest.raises(ValueError, match="Unsupported"):
            tecio.get_zone_list(str(path))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing file raises :exc:`FileNotFoundError` for both functions."""
        with pytest.raises(FileNotFoundError):
            tecio.get_variable_list(str(tmp_path / "ghost.szplt"))
        with pytest.raises(FileNotFoundError):
            tecio.get_zone_list(str(tmp_path / "ghost.dat"))


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

# ===========================================================================
# get_num_zones / get_num_variables / get_title / get_file_type / peek
# ===========================================================================


class TestPeekAndScalarQueries:
    """Tests for ``tecio.get_num_zones``/``get_num_variables``/``get_title``/
    ``get_file_type``/``peek``.
    """

    @pytest.mark.parametrize("fmt", ["szplt", "plt", "dat"])
    def test_scalar_queries_match_full_open(self, fmt: str, tmp_path: Path) -> None:
        """Every scalar query agrees with a full ``tecio.open()`` read."""
        path = tmp_path / f"simple.{fmt}"
        writer = {
            "szplt": _write_simple_szplt,
            "plt": _write_simple_plt,
            "dat": _write_simple_dat,
        }[fmt]
        writer(path, n_zones=3)

        assert tecio.get_num_zones(str(path)) == 3
        assert tecio.get_num_variables(str(path)) == 2

        with tecio.open(str(path), "r") as r:
            assert tecio.get_title(str(path)) == r.title
            assert tecio.get_file_type(str(path)) == r.file_type
            assert tecio.get_num_zones(str(path)) == r.num_zones
            assert tecio.get_num_variables(str(path)) == r.num_vars

    @pytest.mark.parametrize("fmt", ["szplt", "plt", "dat"])
    def test_peek_matches_full_open(self, fmt: str, tmp_path: Path) -> None:
        """``peek()`` agrees with a full ``tecio.open()`` read, every field."""
        path = tmp_path / f"simple.{fmt}"
        writer = {
            "szplt": _write_simple_szplt,
            "plt": _write_simple_plt,
            "dat": _write_simple_dat,
        }[fmt]
        writer(path, n_zones=2)

        summary = tecio.peek(str(path))
        with tecio.open(str(path), "r") as r:
            assert summary.title == r.title
            assert summary.file_type == r.file_type
            assert summary.variable_names == list(r.variables)
            assert [z.title for z in summary.zones] == [z.title for z in r.zones]
            assert [z.zone_type for z in summary.zones] == [
                z.zone_type for z in r.zones
            ]
            assert [z.num_nodes for z in summary.zones] == [
                z.num_nodes for z in r.zones
            ]
            assert [z.num_elements for z in summary.zones] == [
                z.num_elements for z in r.zones
            ]

    def test_peek_dat_fe_zone_dimensions(self, tmp_path: Path) -> None:
        """An FE zone's node/element counts come straight from the header."""
        path = tmp_path / "fe.dat"
        path.write_text(
            'TITLE="t"\nVARIABLES="x" "y"\n'
            'ZONE T="quad", ZONETYPE=FEQUADRILATERAL, NODES=4, ELEMENTS=1, '
            "DATAPACKING=POINT\n"
            "0.0 0.0\n1.0 0.0\n1.0 1.0\n0.0 1.0\n1 2 3 4\n"
        )
        summary = tecio.peek(str(path))
        assert len(summary.zones) == 1
        z = summary.zones[0]
        assert z.title == "quad"
        assert z.zone_type == ZoneType.FEQUADRILATERAL
        assert z.num_nodes == 4
        assert z.num_elements == 1

    def test_peek_dat_no_ijk_dimensions_are_none(self, tmp_path: Path) -> None:
        """An ordered zone with no I/J/K reports unknown dimensions, not a guess."""
        path = tmp_path / "no_ijk.dat"
        path.write_text(
            'TITLE="t"\nVARIABLES="x" "y"\nZONE T="z", F=point\n1.0 10.0\n2.0 20.0\n'
        )
        summary = tecio.peek(str(path))
        assert summary.zones[0].num_nodes is None
        assert summary.zones[0].num_elements is None

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """An unrecognised extension raises :exc:`ValueError` for every query."""
        path = tmp_path / "data.nc"
        path.touch()
        for fn in (
            tecio.get_num_zones,
            tecio.get_num_variables,
            tecio.get_title,
            tecio.get_file_type,
            tecio.peek,
        ):
            with pytest.raises(ValueError, match="Unsupported"):
                fn(str(path))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing file raises :exc:`FileNotFoundError` for every query."""
        ghost = tmp_path / "ghost.szplt"
        for fn in (
            tecio.get_num_zones,
            tecio.get_num_variables,
            tecio.get_title,
            tecio.get_file_type,
            tecio.peek,
        ):
            with pytest.raises(FileNotFoundError):
                fn(str(ghost))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
