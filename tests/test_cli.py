#!/usr/bin/env python3
"""pytest test suite for all tecio CLI tools.

Onera test file contents (confirmed via tecdump):
    Zones  : 2
        Zone 1 — FluidVolume   FEBRICK        46 417 nodes / 43 008 elements
        Zone 2 — WingSurface   FEQUADRILATERAL  1 453 nodes /  1 408 elements
    Variables : 18
        1 x, 2 y, 3 z, 4 Density, 5 Momentum U, 6 Momentum V, 7 Momentum W,
        8 Energy, 9 SA Turbulent Eddy Viscosity, 10 Pressure, 11 Temperature,
        12 Pressure_Coefficient, 13 Mach, 14 Laminar_Viscosity,
        15 Skin_Friction_Coefficient, 16 Heat_Flux, 17 Y_Plus, 18 Eddy_Viscosity
    Both zones: static (strand 0, solution_time 0.0), all variables NODAL FLOAT.
    Zone 2 has zone-level aux data.

Design notes:
    - All ``main()`` functions return int. All error-path assertions use
      ``assert ret == 1`` — no ``pytest.raises(SystemExit)`` needed.
      (Requires the tecscale and tecmerge patches that replace sys.exit calls
      in helper functions with return-None / raise-FileNotFoundError.)
    - File-writing tests use pytest's ``tmp_path`` fixture; the ``tests/``
      directory is never modified.
    - ``tecstats --csv`` auto-names the CSV next to the input, so those tests
      copy the source file into ``tmp_path`` first.
    - The ``onera_path`` fixture is parametrised over all three formats;
      single-format tests reference ``_ONERA["szplt"]`` directly.
    - Both Onera zones are FE, so ``tecslice`` IJK flags produce verbatim
      copies with a warning. Tests verify the warning and zone count.

Test files required in ``tests/``:
    Onera.szplt  Onera.plt  Onera.dat
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tecio
from tecio import ZoneType
from tecio.cli.tec2mat import main as tec2mat
from tecio.cli.tecaux import main as tecaux
from tecio.cli.tecdump import main as tecdump
from tecio.cli.tecextract import main as tecextract
from tecio.cli.tecfix import main as tecfix
from tecio.cli.tecmerge import main as tecmerge
from tecio.cli.teconvert import main as teconvert
from tecio.cli.tecscale import main as tecscale
from tecio.cli.tecslice import main as tecslice
from tecio.cli.tecstats import main as tecstats

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover -> tec2mat requires scipy
    sio = None

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

_TEST_DIR = Path(tecio.__file__).parent.parent / "tests"
_FORMATS = ["szplt", "plt", "dat"]
_ONERA: dict[str, Path] = {fmt: _TEST_DIR / f"Onera.{fmt}" for fmt in _FORMATS}

_NUM_ZONES = 2
_NUM_VARS = 18

# Per-zone (nodes, elements, nodes_per_cell) -- FluidVolume FEBrick, WingSurface
# FEQuad. Used only by the tec2mat tests below.
_Z1_NODES, _Z1_ELEMS, _Z1_NPC = 46417, 43008, 8
_Z2_NODES, _Z2_ELEMS, _Z2_NPC = 1453, 1408, 4


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _is_readable(path: Path) -> bool:
    """Return True if *path* can be opened as a valid Tecplot file."""
    try:
        with tecio.open(str(path), "r") as r:
            return r.num_vars > 0
    except Exception:
        return False


def _copy_onera(src_fmt: str, dst_dir: Path, name: str) -> Path:
    """Copy an Onera file into *dst_dir* under *name*, preserving extension."""
    src = _ONERA[src_fmt]
    dst = dst_dir / f"{name}{src.suffix}"
    shutil.copy(src, dst)
    return dst


def _load_mat(path: Path, *, squeeze: bool = True) -> dict[str, Any]:
    """Load a ``.mat`` file as MATLAB structs (``mat_struct`` objects).

    Used only by the tec2mat tests.
    """
    return sio.loadmat(str(path), struct_as_record=False, squeeze_me=squeeze)


def _struct(container: dict[str, Any], name: str) -> Any:
    """Return struct *name*, unwrapping the 1x1 object array left by squeeze=False.

    Used only by the tec2mat tests.
    """
    obj = container[name]
    if isinstance(obj, np.ndarray):
        obj = obj.reshape(-1)[0]
    return obj


def _write_synthetic_ijk(path: Path) -> None:
    """Write a small 2-zone ASCII DAT exercising passive and shared variables.

    Zone 1 (``Z1``) has all three variables active. Zone 2 (``Z2``) shares
    variable 1 from zone 1, marks variable 2 passive, and keeps variable 3
    active. Each active array has four values.

    Ordered (IJK) zones have no connectivity, so this only exercises
    variable sharing -- see ``shared_dataset``/``shared_path`` above for FE
    zones with connectivity sharing too.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    p = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    p2 = np.array([20.0, 21.0, 22.0, 23.0], dtype=np.float64)
    with tecio.open(str(path), "w", title="syn", variables=["x", "y", "p"]) as w:
        w.write_ordered_zone(data=[x, y, p], title="Z1")
        w.write_ordered_zone(
            data=[p2],
            title="Z2",
            passive_vars={"y"},
            var_sharing={"x": 0},
        )


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def meta() -> dict:
    """Dataset metadata from Onera.szplt, read once per test session."""
    with tecio.open(str(_ONERA["szplt"]), "r") as r:
        return {
            "num_zones": r.num_zones,
            "num_vars": r.num_vars,
            "variables": r.variables,
        }


@pytest.fixture(params=_FORMATS)
def onera_path(request) -> Path:
    """Yield the Onera input file for each supported format."""
    return _ONERA[request.param]


# --------------------------------------------------------------------------------------
# Shared-data fixture
# --------------------------------------------------------------------------------------


def _write_shared_dataset(path: Path) -> None:
    """Write a 3-zone FETETRAHEDRON dataset with real variable/connectivity sharing.

    Zone 0 (t=0.0): owns x, y, z, connectivity, c, and w.
    Zone 1 (t=1.0): shares x, y, z, connectivity, and w from zone 0; c is its own.
    Zone 2 (t=2.0): shares x, y, z, and connectivity from zone 0, and shares c from zone
                    1.
    """
    x = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    z = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    nodes = np.array([[0, 1, 2, 3]], dtype=np.int64)
    c1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    c2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    w1 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    w3 = np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float64)

    with tecio.open(str(path), "w", variables=["x", "y", "z", "c", "w"]) as w:
        w.write_fe_zone(
            zone_type=ZoneType.FETETRAHEDRON,
            data=[x, y, z, c1, w1],
            node_map=nodes,
            title="Zone1_Owner",
            strand_id=1,
            solution_time=0.0,
        )
        w.write_fe_zone(
            zone_type=ZoneType.FETETRAHEDRON,
            data=[c2],
            var_sharing={"x": 0, "y": 0, "z": 0, "w": 0},
            con_sharing=0,
            title="Zone2_SharesFromZone1",
            strand_id=1,
            solution_time=1.0,
        )
        w.write_fe_zone(
            zone_type=ZoneType.FETETRAHEDRON,
            data=[w3],
            var_sharing={"x": 0, "y": 0, "z": 0, "c": 1},
            con_sharing=0,
            title="Zone3_SharesFromZone1And2",
            strand_id=1,
            solution_time=2.0,
        )


@pytest.fixture(scope="session")
def shared_dataset(tmp_path_factory) -> dict[str, Path]:
    """Write the shared dataset once per session, in every format."""
    out_dir = tmp_path_factory.mktemp("shared_dataset")
    paths: dict[str, Path] = {}
    for fmt, ext in zip(_FORMATS, ["szplt", "plt", "dat"], strict=True):
        p = out_dir / f"shared.{ext}"
        _write_shared_dataset(p)
        paths[fmt] = p
    return paths


@pytest.fixture(params=_FORMATS)
def shared_path(request, shared_dataset: dict[str, Path]) -> Path:
    """Yield the shared-dataset input file for each supported format."""
    return shared_dataset[request.param]


def _write_aux_dataset(path: Path) -> None:
    """Same 3-zone sharing layout as ``_write_shared_dataset``, plus real
    auxiliary data at all three levels (dataset, two of three zones, two of
    five variables -- deliberately sparse, so the sparse-export behavior of
    ``--export-json`` gets exercised too).  ``shared_dataset`` is used
    everywhere else specifically *because* it has no pre-existing aux data,
    so this is a separate fixture rather than a modification of it.
    """
    x = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    z = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    nodes = np.array([[0, 1, 2, 3]], dtype=np.int64)
    c1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    c2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    w1 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    w3 = np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float64)

    with tecio.open(str(path), "w", variables=["x", "y", "z", "c", "w"]) as w:
        w.add_auxdataset_dict({"Solver": "MyCFD", "Version": "2.1"})
        w.add_auxvar_dict({0: {"Units": "m"}, 3: {"Units": "Pa"}})
        w.flush_aux()
        w.write_fe_zone(
            zone_type=ZoneType.FETETRAHEDRON,
            data=[x, y, z, c1, w1],
            node_map=nodes,
            title="Zone1_Owner",
            strand_id=1,
            solution_time=0.0,
            aux={"Description": "Wing"},
        )
        w.write_fe_zone(
            zone_type=ZoneType.FETETRAHEDRON,
            data=[c2],
            var_sharing={"x": 0, "y": 0, "z": 0, "w": 0},
            con_sharing=0,
            title="Zone2_SharesFromZone1",
            strand_id=1,
            solution_time=1.0,
        )
        w.write_fe_zone(
            zone_type=ZoneType.FETETRAHEDRON,
            data=[w3],
            var_sharing={"x": 0, "y": 0, "z": 0, "c": 1},
            con_sharing=0,
            title="Zone3_SharesFromZone1And2",
            strand_id=1,
            solution_time=2.0,
            aux={"Description": "Empennage"},
        )


@pytest.fixture(scope="session")
def aux_dataset(tmp_path_factory) -> dict[str, Path]:
    """Write the aux-populated dataset once per session, in every format."""
    out_dir = tmp_path_factory.mktemp("aux_dataset")
    paths: dict[str, Path] = {}
    for fmt, ext in zip(_FORMATS, ["szplt", "plt", "dat"], strict=True):
        p = out_dir / f"aux.{ext}"
        _write_aux_dataset(p)
        paths[fmt] = p
    return paths


@pytest.fixture(params=_FORMATS)
def aux_path(request, aux_dataset: dict[str, Path]) -> Path:
    """Yield the aux-populated dataset's input file for each supported format."""
    return aux_dataset[request.param]


_FLAG_TO_EXT = {"-szplt": ".szplt", "-plt": ".plt", "-dat": ".dat"}


def _other_format_flag(src: Path) -> str:
    """Return a teconvert format flag guaranteed to differ from *src*'s own format.

    ``shared_path`` is parametrized over all three formats, so a fixed
    target flag would silently no-op (and write nothing at all) whenever
    the source happens to already be that format.
    """
    return {".szplt": "-plt", ".plt": "-dat", ".dat": "-szplt"}[src.suffix]


# ======================================================================================
# tecdump
# ======================================================================================


class TestTecdump:
    """Tests for tecdump - prints Tecplot file contents to stdout."""

    def test_basic_output(self, onera_path: Path, capsys) -> None:
        """File header fields appear in stdout; return code is 0."""
        ret = tecdump([str(onera_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "File Type" in out
        assert "Dataset Title" in out
        assert "Num Vars" in out
        assert "Num Zones" in out

    def test_zone_record_present(self, onera_path: Path, capsys) -> None:
        """Zone titles FluidVolume and WingSurface appear in the output."""
        ret = tecdump([str(onera_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "FluidVolume" in out
        assert "WingSurface" in out

    def test_ignore_zones_flag(self, onera_path: Path, capsys) -> None:
        """--ignore-zones suppresses zone records."""
        ret = tecdump(["--ignore-zones", str(onera_path)])
        assert ret == 0
        assert "Zone Record" not in capsys.readouterr().out

    def test_ignore_variables_flag(self, onera_path: Path, capsys) -> None:
        """--ignore-variables suppresses variable value output."""
        ret = tecdump(["--ignore-variables", str(onera_path)])
        assert ret == 0
        assert "Values" not in capsys.readouterr().out

    def test_zone_filter(self, onera_path: Path) -> None:
        """-z 1 limits output to zone 1 without error."""
        assert tecdump(["-z", "1", str(onera_path)]) == 0

    def test_variable_filter(self, onera_path: Path) -> None:
        """-v 1 limits output to variable 1 without error."""
        assert tecdump(["-v", "1", str(onera_path)]) == 0

    def test_maxvals(self, onera_path: Path) -> None:
        """--maxvals changes the array truncation threshold without error."""
        assert tecdump(["--maxvals", "5", str(onera_path)]) == 0


# ======================================================================================
# teconvert
# ======================================================================================


class TestTeconvert:
    """Tests for teconvert - converts between SZL, PLT, and DAT."""

    @pytest.mark.parametrize(
        "src_fmt,dst_flag,dst_ext",
        [
            ("szplt", "-dat", ".dat"),
            ("szplt", "-plt", ".plt"),
            ("plt", "-dat", ".dat"),
            ("plt", "-szplt", ".szplt"),
            ("dat", "-szplt", ".szplt"),
            ("dat", "-plt", ".plt"),
        ],
    )
    def test_convert_between_formats(
        self, src_fmt: str, dst_flag: str, dst_ext: str, tmp_path: Path
    ) -> None:
        """Convert between every supported format pair; output must be readable."""
        dst = tmp_path / f"out{dst_ext}"
        ret = teconvert([
            dst_flag,
            "--force",
            "-o",
            str(dst),
            str(_ONERA[src_fmt]),
        ])
        assert ret == 0
        assert dst.exists()
        assert dst.stat().st_size > 0
        assert _is_readable(dst)

    @pytest.mark.parametrize(
        "fmt,flag",
        [
            ("szplt", "-szplt"),
            ("plt", "-plt"),
            ("dat", "-dat"),
        ],
    )
    def test_same_format_noop(self, fmt: str, flag: str) -> None:
        """Same-format conversion warns and returns 0."""
        ret = teconvert([flag, str(_ONERA[fmt])])
        assert ret == 0

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert teconvert(["-dat", str(tmp_path / "ghost.szplt")]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "out.dat"
        dst.touch()
        assert teconvert(["-dat", "-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "out.dat"
        dst.touch()
        ret = teconvert(["-dat", "--force", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert dst.stat().st_size > 0


# ======================================================================================
# tecextract
# ======================================================================================


class TestTecextract:
    """Tests for tecextract - extracts a subset of zones and/or variables."""

    def test_extract_all(self, onera_path: Path, tmp_path: Path) -> None:
        """No filter produces a readable verbatim copy."""
        dst = tmp_path / "out.szplt"
        assert tecextract(["-o", str(dst), "--force", str(onera_path)]) == 0
        assert _is_readable(dst)

    def test_extract_zone_0_only(self, onera_path: Path, tmp_path: Path) -> None:
        """``-z 0`` extracts FluidVolume; output has 1 zone."""
        dst = tmp_path / "out.szplt"
        ret = tecextract(["-z", "0", "-o", str(dst), str(onera_path)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 1

    def test_extract_zone_1_only(self, onera_path: Path, tmp_path: Path) -> None:
        """``-z 1`` extracts WingSurface; output has 1 zone."""
        dst = tmp_path / "out.szplt"
        ret = tecextract(["-z", "1", "-o", str(dst), str(onera_path)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 1

    def test_extract_variable_0_reduces_count(self, tmp_path: Path) -> None:
        """Output is written to DAT format because the SZL C library requires
        variables 0, 1, and 2 (x, y, z) to be present to compute its
        subzone layout — writing a single-variable SZL file is not supported.
        """
        dst = tmp_path / "out.dat"
        ret = tecextract(["-v", "0", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_vars == 1
            assert r.variables == ["x"]

    def test_extract_variable_0_szplt_requires_xyz(self, tmp_path: Path) -> None:
        """Extracting a single variable to SZL fails — the format requires
        variables 0, 1, and 2 (x, y, z) to compute its subzone layout.
        """
        dst = tmp_path / "out.szplt"
        ret = tecextract(["-v", "0", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 1

    def test_extract_multiple_variables(self, tmp_path: Path) -> None:
        """``-v 0,1,2`` produces a file with exactly 3 variables."""
        dst = tmp_path / "out.szplt"
        ret = tecextract([
            "-v",
            "0,1,2",
            "-o",
            str(dst),
            str(_ONERA["szplt"]),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_vars == 3
            assert r.variables == ["x", "y", "z"]

    def test_extract_zone_and_variable(self, tmp_path: Path) -> None:
        """Combined zone + variable filter applies both restrictions."""
        dst = tmp_path / "out.szplt"
        ret = tecextract([
            "-z",
            "1",
            "-v",
            "9",
            "-o",
            str(dst),
            str(_ONERA["szplt"]),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 1
            assert r.num_vars == 1
            assert r.variables == ["Pressure"]

    def test_output_format_controlled_by_extension(self, tmp_path: Path) -> None:
        """Writing to .dat produces a readable ASCII output."""
        dst = tmp_path / "out.dat"
        assert tecextract(["-o", str(dst), "--force", str(_ONERA["szplt"])]) == 0
        assert _is_readable(dst)

    def test_zone_out_of_range_returns_1(self, tmp_path: Path) -> None:
        """Zone index beyond num_zones (2) returns exit code 1."""
        assert (
            tecextract([
                "-z",
                "99",
                "-o",
                str(tmp_path / "out.szplt"),
                str(_ONERA["szplt"]),
            ])
            == 1
        )

    def test_variable_out_of_range_returns_1(self, tmp_path: Path) -> None:
        """Variable index beyond num_vars (18) returns exit code 1."""
        assert (
            tecextract([
                "-v",
                "99",
                "-o",
                str(tmp_path / "out.szplt"),
                str(_ONERA["szplt"]),
            ])
            == 1
        )

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert (
            tecextract([
                "-o",
                str(tmp_path / "out.szplt"),
                str(tmp_path / "ghost.szplt"),
            ])
            == 1
        )

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "out.szplt"
        dst.touch()
        assert tecextract(["-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "out.szplt"
        dst.touch()
        ret = tecextract(["--force", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert dst.stat().st_size > 0


# ======================================================================================
# tecfix
# ======================================================================================


class TestTecfix:
    """Tests for tecfix — marks NaN / Inf variable arrays as passive."""

    def test_clean_file_produces_copy(
        self, onera_path: Path, tmp_path: Path, capsys
    ) -> None:
        """A file with no NaN/Inf yields a clean copy and a 'clean' message."""
        dst = tmp_path / "fixed.szplt"
        ret = tecfix(["-o", str(dst), "--force", str(onera_path)])
        assert ret == 0
        assert _is_readable(dst)
        assert "clean" in capsys.readouterr().out.lower()

    def test_clean_copy_preserves_structure(self, tmp_path: Path) -> None:
        """Fixed file has the same zone and variable counts as the source."""
        dst = tmp_path / "fixed.szplt"
        tecfix(["--force", "-o", str(dst), str(_ONERA["szplt"])])
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES
            assert r.num_vars == _NUM_VARS

    def test_dry_run_does_not_write(self, onera_path: Path, tmp_path: Path) -> None:
        """--dry-run scans for issues without creating an output file."""
        dst = tmp_path / "fixed.szplt"
        assert tecfix(["--dry-run", str(onera_path)]) == 0
        assert not dst.exists()

    def test_dry_run_clean_message(self, onera_path: Path, capsys) -> None:
        """--dry-run on a clean file reports it is clean."""
        tecfix(["--dry-run", str(onera_path)])
        assert "clean" in capsys.readouterr().out.lower()

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert tecfix([str(tmp_path / "ghost.szplt")]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "fixed.szplt"
        dst.touch()
        assert tecfix(["-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "fixed.szplt"
        dst.touch()
        ret = tecfix(["--force", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert dst.stat().st_size > 0


# ======================================================================================
# tecmerge
# ======================================================================================


class TestTecmerge:
    """Tests for tecmerge - merges zones from multiple files into one.

    Design note on deduplication:
        tecmerge._expand_inputs deduplicates resolved paths, so passing the
        same path twice is equivalent to passing it once — intended behaviour
        for glob patterns. Tests that need N distinct "files" therefore copy
        the source into tmp_path under different names so each resolves to a
        unique path.
    """

    def test_merge_two_distinct_files_doubles_zones(self, tmp_path: Path) -> None:
        """Merging two distinct copies produces a file with 4 zones (2 × 2)."""
        src1 = _copy_onera("szplt", tmp_path, "a")
        src2 = _copy_onera("dat", tmp_path, "b")
        dst = tmp_path / "merged.szplt"
        ret = tecmerge(["-o", str(dst), str(src1), str(src2)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES * 2
            assert r.num_vars == _NUM_VARS

    def test_merge_mixed_formats(self, tmp_path: Path) -> None:
        """SZL + DAT inputs are merged into a single readable output."""
        dst = tmp_path / "merged.szplt"
        ret = tecmerge(["-o", str(dst), str(_ONERA["szplt"]), str(_ONERA["dat"])])
        assert ret == 0
        assert _is_readable(dst)

    def test_merge_three_formats(self, tmp_path: Path) -> None:
        """Three distinct copies produce 6 zones (3 × 2)."""
        src1 = _copy_onera("szplt", tmp_path, "a")
        src2 = _copy_onera("dat", tmp_path, "b")
        src3 = _copy_onera("szplt", tmp_path, "c")
        dst = tmp_path / "merged.szplt"
        ret = tecmerge(["-o", str(dst), str(src1), str(src2), str(src3)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES * 3

    def test_assign_time_strands_with_delta(self, tmp_path: Path) -> None:
        """``-d``/``--delta`` assigns evenly-spaced solution times per input file."""
        src1 = _copy_onera("szplt", tmp_path, "t0")
        src2 = _copy_onera("dat", tmp_path, "t1")
        dst = tmp_path / "transient.szplt"
        ret = tecmerge([
            "--assign-time-strands",
            "-s",
            "0.0",
            "-d",
            "1.0",
            "--strand",
            "1",
            "-o",
            str(dst),
            str(src1),
            str(src2),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            # Zones 0..N-1 from file 0 → time 0.0; zones N..2N-1 from file 1 → 1.0
            assert r.zones[0].solution_time == pytest.approx(0.0)
            assert r.zones[_NUM_ZONES].solution_time == pytest.approx(1.0)

    def test_assign_time_strands_with_end(self, tmp_path: Path) -> None:
        """-e/--end computes the step size automatically from start/end/N."""
        src1 = _copy_onera("szplt", tmp_path, "t0")
        src2 = _copy_onera("dat", tmp_path, "t1")
        src3 = _copy_onera("szplt", tmp_path, "t2")
        dst = tmp_path / "transient_end.szplt"
        ret = tecmerge([
            "--assign-time-strands",
            "-s",
            "0.0",
            "-e",
            "4.0",
            "--strand",
            "1",
            "-o",
            str(dst),
            str(src1),
            str(src2),
            str(src3),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.zones[0].solution_time == pytest.approx(0.0)
            assert r.zones[_NUM_ZONES * 2].solution_time == pytest.approx(4.0)

    def test_strand_offset_shifts_per_zone_position_strand(
        self, tmp_path: Path
    ) -> None:
        """--strand-offset shifts the default per-zone-position strand IDs."""
        src1 = _copy_onera("szplt", tmp_path, "t0")
        src2 = _copy_onera("dat", tmp_path, "t1")
        dst = tmp_path / "transient_offset.szplt"
        ret = tecmerge([
            "--assign-time-strands",
            "-s",
            "0.0",
            "-d",
            "1.0",
            "--strand-offset",
            "100",
            "-o",
            str(dst),
            str(src1),
            str(src2),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            # Without an offset, zones within each file get strand 1, 2, ...;
            # with --strand-offset 100 they become 101, 102, ...
            for zi in range(_NUM_ZONES):
                assert r.zones[zi].strand_id == zi + 1 + 100
                assert r.zones[_NUM_ZONES + zi].strand_id == zi + 1 + 100

    def test_strand_offset_shifts_fixed_strand(self, tmp_path: Path) -> None:
        """--strand-offset also shifts a --strand override, not just the default."""
        src1 = _copy_onera("szplt", tmp_path, "t0")
        src2 = _copy_onera("dat", tmp_path, "t1")
        dst = tmp_path / "transient_fixed_offset.szplt"
        ret = tecmerge([
            "--assign-time-strands",
            "-s",
            "0.0",
            "-d",
            "1.0",
            "--strand",
            "1",
            "--strand-offset",
            "100",
            "-o",
            str(dst),
            str(src1),
            str(src2),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            for zone in r.zones:
                assert zone.strand_id == 101

    def test_missing_start_with_assign_ts_returns_1(self, tmp_path: Path) -> None:
        """--assign-time-strands without -s/--start returns exit code 1."""
        assert (
            tecmerge([
                "--assign-time-strands",
                "-d",
                "1.0",
                "-o",
                str(tmp_path / "out.szplt"),
                str(_ONERA["szplt"]),
            ])
            == 1
        )

    def test_missing_delta_and_end_returns_1(self, tmp_path: Path) -> None:
        """--assign-time-strands without -d/--delta or -e/--end returns exit code 1."""
        assert (
            tecmerge([
                "--assign-time-strands",
                "-s",
                "0.0",
                "-o",
                str(tmp_path / "out.szplt"),
                str(_ONERA["szplt"]),
            ])
            == 1
        )

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent input file returns exit code 1."""
        assert (
            tecmerge([
                "-o",
                str(tmp_path / "out.szplt"),
                str(tmp_path / "ghost.szplt"),
            ])
            == 1
        )

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "merged.szplt"
        dst.touch()
        assert (
            tecmerge([
                "-o",
                str(dst),
                str(_ONERA["szplt"]),
                str(_ONERA["dat"]),
            ])
            == 1
        )

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "merged.szplt"
        dst.touch()
        ret = tecmerge([
            "--force",
            "-o",
            str(dst),
            str(_ONERA["szplt"]),
            str(_ONERA["dat"]),
        ])
        assert ret == 0
        assert dst.stat().st_size > 0


# ======================================================================================
# tecscale
# ======================================================================================


class TestTecscale:
    """Tests for tecscale - scales and/or offsets one or more variables."""

    @staticmethod
    def _read(path: Path, zone: int, var: int) -> np.ndarray:
        """Return one variable of one zone as a flat float64 array."""
        with tecio.open(str(path), "r") as r:
            return r.zones[zone].variables[var].values.ravel().astype(np.float64)

    def test_scale_by_index(self, onera_path: Path, tmp_path: Path) -> None:
        """Scale variable 0 (x) by a constant factor; output is readable."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale(["-v", "0", "-s", "2.0", "-o", str(dst), str(onera_path)])
        assert ret == 0
        assert _is_readable(dst)

    def test_scale_by_name(self, onera_path: Path, tmp_path: Path) -> None:
        """Scale the Pressure variable identified by name."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale([
            "-v",
            "Pressure",
            "--scale",
            "1e-3",
            "-o",
            str(dst),
            str(onera_path),
        ])
        assert ret == 0
        assert _is_readable(dst)

    def test_offset_only(self, onera_path: Path, tmp_path: Path) -> None:
        """Offset alone leaves the scale at its default of 1."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale([
            "-v",
            "Temperature",
            "--offset",
            "-273.15",
            "-o",
            str(dst),
            str(onera_path),
        ])
        assert ret == 0
        with tecio.open(str(onera_path), "r") as r:
            names = [v.name for v in r.zones[0].variables]
        t_idx = names.index("Temperature")
        orig = self._read(onera_path, 0, t_idx)
        np.testing.assert_allclose(self._read(dst, 0, t_idx), orig - 273.15, rtol=1e-5)

    def test_scale_with_offset(self, onera_path: Path, tmp_path: Path) -> None:
        """Scale and offset together apply new = old * scale + offset."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale([
            "-v",
            "0",
            "-s",
            "2.0",
            "--offset",
            "5.0",
            "-o",
            str(dst),
            str(onera_path),
        ])
        assert ret == 0
        orig = self._read(onera_path, 0, 0)
        np.testing.assert_allclose(self._read(dst, 0, 0), orig * 2.0 + 5.0, rtol=1e-5)

    def test_offset_before_scale(self, onera_path: Path, tmp_path: Path) -> None:
        """The order of --scale and --offset after -v does not matter."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale([
            "-v",
            "0",
            "--offset",
            "5.0",
            "-s",
            "2.0",
            "-o",
            str(dst),
            str(onera_path),
        ])
        assert ret == 0
        orig = self._read(onera_path, 0, 0)
        np.testing.assert_allclose(self._read(dst, 0, 0), orig * 2.0 + 5.0, rtol=1e-5)

    def test_variable_alone_is_identity(self, onera_path: Path, tmp_path: Path) -> None:
        """A bare -v with no --scale/--offset defaults to scale 1, offset 0."""
        dst = tmp_path / "scaled.szplt"
        assert tecscale(["-v", "0", "-o", str(dst), str(onera_path)]) == 0
        orig = self._read(onera_path, 0, 0)
        np.testing.assert_allclose(self._read(dst, 0, 0), orig, rtol=1e-5)

    def test_scale_single_zone(self, onera_path: Path, tmp_path: Path) -> None:
        """-z restricts scaling to zone 0 only."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale([
            "-v",
            "Density",
            "-s",
            "1000.0",
            "-z",
            "0",
            "-o",
            str(dst),
            str(onera_path),
        ])
        assert ret == 0
        assert dst.exists()

    def test_scale_values_correct(self, tmp_path: Path) -> None:
        """Scaled values in zone 0 equal original x scale_factor."""
        src = _ONERA["szplt"]
        dst = tmp_path / "scaled.szplt"
        scale = 2.0
        tecscale(["-v", "0", "-s", str(scale), "--force", "-o", str(dst), str(src)])
        orig = self._read(src, 0, 0)
        np.testing.assert_allclose(self._read(dst, 0, 0), orig * scale, rtol=1e-5)

    def test_unscaled_zone_unchanged(self, tmp_path: Path) -> None:
        """When -z 0 is active, zone 1 values are identical to source."""
        src = _ONERA["szplt"]
        dst = tmp_path / "scaled.szplt"
        tecscale([
            "-v",
            "Pressure",
            "-s",
            "999.0",
            "-z",
            "0",
            "--force",
            "-o",
            str(dst),
            str(src),
        ])
        np.testing.assert_allclose(
            self._read(dst, 1, 9), self._read(src, 1, 9), rtol=1e-5
        )

    def test_multiple_variables_in_one_pass(self, tmp_path: Path) -> None:
        """Each -v binds the --scale/--offset that follow it, not the others."""
        src = _ONERA["szplt"]
        dst = tmp_path / "scaled.szplt"
        ret = tecscale([
            "-v",
            "0",
            "-s",
            "2.0",
            "-v",
            "Temperature",
            "--offset",
            "-273.15",
            "--force",
            "-o",
            str(dst),
            str(src),
        ])
        assert ret == 0
        with tecio.open(str(src), "r") as r_src:
            names = [v.name for v in r_src.zones[0].variables]
        temp_idx = names.index("Temperature")
        np.testing.assert_allclose(
            self._read(dst, 0, 0), self._read(src, 0, 0) * 2.0, rtol=1e-5
        )
        np.testing.assert_allclose(
            self._read(dst, 0, temp_idx),
            self._read(src, 0, temp_idx) - 273.15,
            rtol=1e-5,
        )

    def test_options_do_not_leak_to_next_variable(self, tmp_path: Path) -> None:
        """A second -v without its own flags falls back to the defaults."""
        src = _ONERA["szplt"]
        dst = tmp_path / "scaled.szplt"
        ret = tecscale([
            "-v",
            "0",
            "-s",
            "2.0",
            "--offset",
            "5.0",
            "-v",
            "1",
            "--force",
            "-o",
            str(dst),
            str(src),
        ])
        assert ret == 0
        np.testing.assert_allclose(
            self._read(dst, 0, 1), self._read(src, 0, 1), rtol=1e-5
        )

    def test_repeated_same_variable_keeps_last(self, tmp_path: Path) -> None:
        """Naming the same variable twice in one call keeps only the last -v."""
        src = _ONERA["szplt"]
        dst = tmp_path / "scaled.szplt"
        tecscale([
            "-v",
            "0",
            "-s",
            "999.0",
            "-v",
            "0",
            "-s",
            "2.0",
            "--force",
            "-o",
            str(dst),
            str(src),
        ])
        np.testing.assert_allclose(
            self._read(dst, 0, 0), self._read(src, 0, 0) * 2.0, rtol=1e-5
        )

    def test_scale_before_variable_errors(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """--scale or --offset with no preceding -v is an argparse error."""
        for flag in ("--scale", "--offset"):
            with pytest.raises(SystemExit) as exc:
                tecscale([
                    flag,
                    "2.0",
                    "-v",
                    "0",
                    "-o",
                    str(tmp_path / "scaled.szplt"),
                    str(onera_path),
                ])
            assert exc.value.code == 2

    def test_missing_variable_errors(self, onera_path: Path, tmp_path: Path) -> None:
        """Omitting -v entirely is an argparse error."""
        with pytest.raises(SystemExit) as exc:
            tecscale(["-s", "2.0", "-o", str(tmp_path / "s.szplt"), str(onera_path)])
        assert exc.value.code == 2

    def test_invalid_variable_name_returns_1(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Unknown variable name returns exit code 1."""
        assert (
            tecscale([
                "-v",
                "DOES_NOT_EXIST_XYZ",
                "-o",
                str(tmp_path / "scaled.szplt"),
                str(onera_path),
            ])
            == 1
        )

    def test_variable_index_out_of_range_returns_1(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Variable index beyond num_vars (18) returns exit code 1."""
        assert (
            tecscale([
                "-v",
                "99",
                "-o",
                str(tmp_path / "scaled.szplt"),
                str(onera_path),
            ])
            == 1
        )

    @pytest.mark.parametrize("flag", ["--scale", "--offset"])
    def test_non_numeric_value_returns_1(
        self, flag: str, onera_path: Path, tmp_path: Path
    ) -> None:
        """A non-numeric --scale/--offset returns exit code 1 rather than raising."""
        assert (
            tecscale([
                "-v",
                "0",
                flag,
                "not-a-number",
                "-o",
                str(tmp_path / "scaled.szplt"),
                str(onera_path),
            ])
            == 1
        )

    def test_zone_out_of_range_returns_1(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Zone index beyond num_zones (2) returns exit code 1."""
        assert (
            tecscale([
                "-v",
                "1",
                "-z",
                "99",
                "-o",
                str(tmp_path / "scaled.szplt"),
                str(onera_path),
            ])
            == 1
        )

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert (
            tecscale([
                "-v",
                "1",
                "-o",
                str(tmp_path / "out.szplt"),
                str(tmp_path / "ghost.szplt"),
            ])
            == 1
        )

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "out.szplt"
        dst.touch()
        assert tecscale(["-v", "1", "-o", str(dst), str(_ONERA["szplt"])]) == 1


# ======================================================================================
# tecslice
# ======================================================================================


class TestTecslice:
    """Tests for tecslice - slices structured zones along IJK or time.

    Both Onera zones are FE (FEBRICK and FEQUADRILATERAL), so IJK slice flags
    produce verbatim copies with a warning to stderr. Tests verify that
    behaviour explicitly.
    """

    def test_ijk_slice_on_fe_copies_verbatim(
        self, onera_path: Path, tmp_path: Path, capsys
    ) -> None:
        """IJK ::2 on FE data returns 0; each zone is copied verbatim."""
        dst = tmp_path / "sliced.szplt"
        ret = tecslice(["-i", "::2", "-o", str(dst), str(onera_path)])
        assert ret == 0
        assert dst.exists()
        assert "unstructured" in capsys.readouterr().err.lower()

    def test_ijk_slice_preserves_all_zones(self, tmp_path: Path) -> None:
        """Verbatim copy retains the full zone and variable counts."""
        dst = tmp_path / "sliced.szplt"
        tecslice(["-i", "::2", "-j", "::2", "-o", str(dst), str(_ONERA["szplt"])])
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES
            assert r.num_vars == _NUM_VARS

    def test_no_flags_warns_and_copies(
        self, onera_path: Path, tmp_path: Path, capsys
    ) -> None:
        """No slice flags prints a warning to stderr and returns 0."""
        dst = tmp_path / "copy.szplt"
        ret = tecslice(["-o", str(dst), str(onera_path)])
        assert ret == 0
        assert dst.exists()
        assert "no slice" in capsys.readouterr().err.lower()

    def test_time_filter_on_static_keeps_all_zones(self, tmp_path: Path) -> None:
        """Static zones (strand 0) are always kept regardless of time filter."""
        dst = tmp_path / "sliced.szplt"
        ret = tecslice(["-t", "0.5:1.0", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES

    def test_strand_filter_on_static_keeps_all_zones(self, tmp_path: Path) -> None:
        """--strand-id on a static file keeps all zones (strand 0 is immune)."""
        dst = tmp_path / "sliced.szplt"
        ret = tecslice([
            "--strand-id",
            "1",
            "-t",
            "0.0:10.0",
            "-o",
            str(dst),
            str(_ONERA["szplt"]),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES

    def test_output_format_controlled_by_extension(self, tmp_path: Path) -> None:
        """Output extension controls the file format."""
        dst = tmp_path / "sliced.dat"
        ret = tecslice(["-i", "::2", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert _is_readable(dst)

    def test_invalid_time_skip_returns_1(self, tmp_path: Path) -> None:
        """Negative time skip (::-1) returns exit code 1."""
        assert (
            tecslice([
                "-t",
                "::-1",
                "-o",
                str(tmp_path / "out.szplt"),
                str(_ONERA["szplt"]),
            ])
            == 1
        )

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert (
            tecslice([
                "-i",
                "::2",
                "-o",
                str(tmp_path / "sliced.szplt"),
                str(tmp_path / "ghost.szplt"),
            ])
            == 1
        )

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "sliced.szplt"
        dst.touch()
        assert tecslice(["-i", "::2", "-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "sliced.szplt"
        dst.touch()
        ret = tecslice([
            "--force",
            "-i",
            "::2",
            "-o",
            str(dst),
            str(_ONERA["szplt"]),
        ])
        assert ret == 0
        assert dst.stat().st_size > 0


# ======================================================================================
# tecstats
# ======================================================================================


class TestTecstats:
    """Tests for tecstats - prints and optionally exports variable statistics."""

    def test_basic_console_output(self, onera_path: Path, capsys) -> None:
        """Statistics table is printed with the expected headers; returns 0."""
        ret = tecstats([str(onera_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Zone" in out
        assert "Zone Title" in out
        assert "Min" in out
        assert "Max" in out

    def test_zone_titles_in_output(self, onera_path: Path, capsys) -> None:
        """Zone titles FluidVolume and WingSurface appear in the table."""
        tecstats([str(onera_path)])
        out = capsys.readouterr().out
        assert "FluidVolume" in out
        assert "WingSurface" in out

    def test_zone_filter(self, onera_path: Path) -> None:
        """-z 1 runs without error."""
        assert tecstats(["-z", "1", str(onera_path)]) == 0

    def test_variable_filter(self, onera_path: Path) -> None:
        """-v 9 (Pressure) runs without error."""
        assert tecstats(["-v", "9", str(onera_path)]) == 0

    def test_zone_and_variable_filter(self, onera_path: Path) -> None:
        """Combined -z 1 -v 9 runs without error."""
        assert tecstats(["-z", "1", "-v", "9", str(onera_path)]) == 0

    def test_csv_created_with_stats_suffix(self, tmp_path: Path) -> None:
        """--csv creates <stem>_stats.csv next to the input file."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        assert tecstats(["--csv", str(src)]) == 0
        assert (tmp_path / "Onera_stats.csv").exists()

    def test_csv_zone_suffix(self, tmp_path: Path) -> None:
        """--csv -z N produces <stem>_zone_N_stats.csv."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        assert tecstats(["--csv", "-z", "1", str(src)]) == 0
        assert (tmp_path / "Onera_zone_1_stats.csv").exists()

    def test_csv_variable_suffix(self, tmp_path: Path) -> None:
        """--csv -v N produces <stem>_var_N_stats.csv."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        assert tecstats(["--csv", "-v", "9", str(src)]) == 0
        assert (tmp_path / "Onera_var_9_stats.csv").exists()

    def test_csv_zone_and_variable_suffix(self, tmp_path: Path) -> None:
        """Combined filters produce <stem>_zone_N_var_M_stats.csv."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        assert tecstats(["--csv", "-z", "1", "-v", "9", str(src)]) == 0
        assert (tmp_path / "Onera_zone_1_var_9_stats.csv").exists()

    def test_csv_header_and_data_rows(self, tmp_path: Path) -> None:
        """CSV has the correct header and 36 data rows (2 zones × 18 vars)."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        tecstats(["--csv", str(src)])
        lines = (tmp_path / "Onera_stats.csv").read_text(encoding="utf-8").splitlines()
        expected_header = (
            "zone_num,zone_title,var_num,var_name,min,max,mean,std,location,note"
        )
        assert lines[0] == expected_header
        assert len(lines) == _NUM_ZONES * _NUM_VARS + 1  # 36 data rows + header

    def test_csv_zone_title_in_rows(self, tmp_path: Path) -> None:
        """Zone title column contains the expected zone names."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        tecstats(["--csv", str(src)])
        content = (tmp_path / "Onera_stats.csv").read_text(encoding="utf-8")
        assert "FluidVolume" in content
        assert "WingSurface" in content

    def test_csv_existing_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing CSV without --force returns exit code 1."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        (tmp_path / "Onera_stats.csv").touch()
        assert tecstats(["--csv", str(src)]) == 1

    def test_csv_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force replaces an existing CSV with fresh content."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        stale = tmp_path / "Onera_stats.csv"
        stale.write_text("stale", encoding="utf-8")
        ret = tecstats(["--csv", "--force", str(src)])
        assert ret == 0
        assert stale.read_text(encoding="utf-8") != "stale"

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert tecstats([str(tmp_path / "ghost.szplt")]) == 1


# ======================================================================================
# Sharing preservation
# ======================================================================================


def _run_teconvert_other_format(tmp_path: Path, src: Path) -> tuple[Path, int]:
    """Run teconvert to a format guaranteed to differ from *src*'s own."""
    flag = _other_format_flag(src)
    dst = tmp_path / f"copy{_FLAG_TO_EXT[flag]}"
    ret = teconvert([flag, "-o", str(dst), "--force", str(src)])
    return dst, ret


class TestSharingPreservation:
    """Verify variable/connectivity sharing survives each tool correctly.

    Covers both fixed bugs (a shared variable used to come out as an empty
    placeholder in tecextract/tecmerge; tecstats double-offset the zone
    number) and new capability (tecextract/tecmerge now remap sharing to the
    new, compacted zone indices instead of unconditionally dropping it).
    """

    # -- tecextract: source zone included -> remap -------------------------------------

    def test_tecextract_remaps_sharing_when_source_included(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """Extracting all 3 zones preserves sharing at the (unchanged) indices."""
        dst = tmp_path / f"extract_all{shared_path.suffix}"
        ret = tecextract(["-o", str(dst), "--force", str(shared_path)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 3
            assert r.zones[1].variables[0].shared_zone == 0  # zone 1's x <- zone 0
            assert r.zones[1].shared_connectivity == 0
            assert r.zones[2].variables[0].shared_zone == 0  # zone 2's x <- zone 0
            assert r.zones[2].variables[3].shared_zone == 1  # zone 2's c <- zone 1
            np.testing.assert_allclose(
                r.zones[1].variables[0].values.ravel(), [0.0, 1.0, 0.0, 0.0]
            )

    def test_tecextract_remaps_sharing_to_compacted_indices(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """Extracting zones 0 and 2 (dropping 1) remaps zone 2's shares correctly.

        Zone 2 shares x/y/z/connectivity from zone 0 (still present, becomes
        output zone 1) but shares c from zone 1 (excluded) -- so c must fall
        back to real data while x/y/z/connectivity remap to output zone 0.
        """
        dst = tmp_path / f"extract_0_2{shared_path.suffix}"
        ret = tecextract(["-z", "0,2", "-o", str(dst), "--force", str(shared_path)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 2
            # Output zone 1 (was source zone 2): x/y/z/connectivity remap to
            # output zone 0 (was source zone 0).
            assert r.zones[1].variables[0].shared_zone == 0
            assert r.zones[1].shared_connectivity == 0
            # c was shared from zone 1, which isn't in this extraction --
            # must be real, independent data, not an empty placeholder.
            assert r.zones[1].variables[3].shared_zone is None
            c_vals = r.zones[1].variables[3].values
            assert c_vals is not None
            np.testing.assert_allclose(c_vals.ravel(), [5.0, 6.0, 7.0, 8.0])

    def test_tecextract_materializes_when_source_excluded(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """Extracting only zone 1 (source zone 0 excluded) writes real data.

        This is the regression case for the original bug: zone 1's shared
        x/y/z/connectivity must come out as actual values, not an empty
        array that would corrupt or crash the output.
        """
        dst = tmp_path / f"extract_1_only{shared_path.suffix}"
        ret = tecextract(["-z", "1", "-o", str(dst), "--force", str(shared_path)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 1
            zone = r.zones[0]
            assert zone.variables[0].shared_zone is None
            assert zone.shared_connectivity is None
            np.testing.assert_allclose(
                zone.variables[0].values.ravel(), [0.0, 1.0, 0.0, 0.0]
            )
            np.testing.assert_array_equal(zone.node_map, [[0, 1, 2, 3]])

    # -- tecmerge: sharing preserved within each file -> offset for later files --------

    def test_tecmerge_preserves_sharing_per_file(self, tmp_path: Path) -> None:
        """Merging two copies preserves sharing within each file's own block.

        File 2's zones must share from file 2's own (re-offset) zone
        indices, not file 1's -- sharing is always file-local.
        """
        src1 = tmp_path / "a.szplt"
        src2 = tmp_path / "b.szplt"
        _write_shared_dataset(src1)
        _write_shared_dataset(src2)
        dst = tmp_path / "merged.szplt"

        ret = tecmerge(["-o", str(dst), str(src1), str(src2)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 6
            # File 1's block: zones 0-2, unchanged indices.
            assert r.zones[1].variables[0].shared_zone == 0
            assert r.zones[2].variables[3].shared_zone == 1
            # File 2's block: zones 3-5, offset by 3.
            assert r.zones[4].variables[0].shared_zone == 3
            assert r.zones[4].shared_connectivity == 3
            assert r.zones[5].variables[3].shared_zone == 4

    # -- 1:1 zone-copy tools: sharing preserved unchanged ------------------------------

    @pytest.mark.parametrize(
        "run_tool",
        [
            lambda tmp_path, src: (
                tmp_path / "copy.szplt",
                tecfix(["-o", str(tmp_path / "copy.szplt"), "--force", str(src)]),
            ),
            lambda tmp_path, src: (
                tmp_path / "copy.szplt",
                tecscale([
                    "-v",
                    "c",
                    "-s",
                    "1.0",
                    "-o",
                    str(tmp_path / "copy.szplt"),
                    "--force",
                    str(src),
                ]),
            ),
            lambda tmp_path, src: _run_teconvert_other_format(tmp_path, src),
            lambda tmp_path, src: (
                tmp_path / "copy.szplt",
                tecslice(["-o", str(tmp_path / "copy.szplt"), str(src)]),
            ),
        ],
        ids=["tecfix", "tecscale", "teconvert", "tecslice_verbatim"],
    )
    def test_verbatim_copy_tools_preserve_sharing(
        self, run_tool, shared_path: Path, tmp_path: Path
    ) -> None:
        """A tool that copies every zone in order keeps sharing at the same indices."""
        dst, ret = run_tool(tmp_path, shared_path)
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 3
            assert r.zones[1].variables[0].shared_zone == 0
            assert r.zones[1].shared_connectivity == 0
            assert r.zones[2].variables[3].shared_zone == 1
            # Shared values still resolve to the real source data.
            np.testing.assert_allclose(
                r.zones[1].variables[0].values.ravel(), [0.0, 1.0, 0.0, 0.0]
            )

    # -- tecstats: off-by-one fix ------------------------------------------------------

    def test_tecstats_shared_note_zone_number_correct(
        self, shared_path: Path, capsys
    ) -> None:
        """The 'shared (zone N)' note uses the correct 0-based zone number.

        Regression test for a double-offset: shared_zone is already
        0-based, and the note used to add 1 on top of it.
        """
        ret = tecstats([str(shared_path)])
        assert ret == 0
        out = capsys.readouterr().out
        # Zone 1's x is shared from zone 0, and zone 2's c is shared from
        # zone 1 specifically -- both distinct notes should appear exactly,
        # confirming the fix isn't just "any single value happens to look
        # right" (the old +1 bug would have printed "zone 1" and "zone 2"
        # here instead).
        assert "shared (zone 0)" in out
        assert "shared (zone 1)" in out


# ======================================================================================
# tec2mat
# ======================================================================================


@pytest.mark.skipif(sio is None, reason="scipy is required by tec2mat itself")
class TestTec2mat:
    """Tests for tec2mat - converts a Tecplot file to a MATLAB .mat file."""

    # -- File naming and basic output --------------------------------------------------

    def test_default_output_name(self, onera_path: Path, tmp_path: Path) -> None:
        """With no -o, output is <stem>.mat next to the input file."""
        src = tmp_path / onera_path.name
        shutil.copy(onera_path, src)
        assert tec2mat([str(src)]) == 0
        assert (tmp_path / "Onera.mat").exists()

    def test_explicit_output_path(self, onera_path: Path, tmp_path: Path) -> None:
        """-o writes the .mat to the requested path."""
        dst = tmp_path / "out.mat"
        assert tec2mat(["-o", str(dst), str(onera_path)]) == 0
        assert dst.exists()
        assert dst.stat().st_size > 0

    # -- Info struct -------------------------------------------------------------------

    def test_info_struct(self, onera_path: Path, tmp_path: Path) -> None:
        """The info struct mirrors the dataset title, counts, and variable names."""
        dst = tmp_path / "out.mat"
        assert tec2mat(["-o", str(dst), str(onera_path)]) == 0
        info = _struct(_load_mat(dst), "info")
        assert int(info.num_zones) == _NUM_ZONES
        assert int(info.num_vars) == _NUM_VARS
        names = [str(v) for v in np.atleast_1d(info.var_names)]
        with tecio.open(str(onera_path), "r") as r:
            assert str(info.title) == r.title
            assert str(info.file_type) == r.file_type.name
            assert names == r.variables

    # -- Zone structs ------------------------------------------------------------------

    def test_zone_structs_present(self, onera_path: Path, tmp_path: Path) -> None:
        """One struct per zone is emitted, 1-based, with no extras."""
        dst = tmp_path / "out.mat"
        tec2mat(["-o", str(dst), str(onera_path)])
        d = _load_mat(dst)
        assert "zone_1" in d
        assert "zone_2" in d
        assert "zone_3" not in d

    def test_zone_metadata(self, onera_path: Path, tmp_path: Path) -> None:
        """Zone titles, types, and node/element counts are preserved."""
        dst = tmp_path / "out.mat"
        tec2mat(["-o", str(dst), str(onera_path)])
        d = _load_mat(dst)
        z1 = _struct(d, "zone_1")
        z2 = _struct(d, "zone_2")
        assert str(z1.title) == "FluidVolume"
        assert str(z1.zone_type) == "FEBRICK"
        assert int(z1.num_nodes) == _Z1_NODES
        assert int(z1.num_elements) == _Z1_ELEMS
        assert str(z2.title) == "WingSurface"
        assert str(z2.zone_type) == "FEQUADRILATERAL"
        assert int(z2.num_nodes) == _Z2_NODES
        assert int(z2.num_elements) == _Z2_ELEMS

    def test_node_map_for_fe_zones(self, onera_path: Path, tmp_path: Path) -> None:
        """Both FE zones carry a 1-based (num_elements x nodes_per_cell) node map."""
        dst = tmp_path / "out.mat"
        tec2mat(["-o", str(dst), str(onera_path)])
        d = _load_mat(dst)
        z1 = _struct(d, "zone_1")
        z2 = _struct(d, "zone_2")
        assert z1.node_map.shape == (_Z1_ELEMS, _Z1_NPC)
        assert z2.node_map.shape == (_Z2_ELEMS, _Z2_NPC)
        assert int(z1.node_map.min()) >= 1
        assert int(z2.node_map.min()) >= 1

    def test_variable_arrays_match_node_count(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Every active nodal variable has one value per node."""
        dst = tmp_path / "out.mat"
        tec2mat(["-o", str(dst), str(onera_path)])
        z1 = _struct(_load_mat(dst), "zone_1")
        for k in range(1, _NUM_VARS + 1):
            assert getattr(z1, f"var_{k}").size == _Z1_NODES

    def test_metadata_arrays_active_nodal(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Onera variables are all active, nodal, and unshared."""
        dst = tmp_path / "out.mat"
        tec2mat(["-o", str(dst), str(onera_path)])
        z1 = _struct(_load_mat(dst), "zone_1")
        assert [str(s) for s in np.atleast_1d(z1.var_status)] == ["active"] * _NUM_VARS
        assert [str(s) for s in np.atleast_1d(z1.var_locations)] == [
            "NODAL"
        ] * _NUM_VARS
        assert np.array_equal(np.atleast_1d(z1.var_shared_from), [0] * _NUM_VARS)

    # -- dtype preservation ------------------------------------------------------------

    def test_dtype_preserved(self, onera_path: Path, tmp_path: Path) -> None:
        """Each stored array keeps the reader's on-disk dtype (DAT=double, PLT/SZL=single)."""  # noqa: E501
        dst = tmp_path / "out.mat"
        tec2mat(["-o", str(dst), str(onera_path)])
        z1 = _struct(_load_mat(dst), "zone_1")
        with tecio.open(str(onera_path), "r") as r:
            zone0 = r.zones[0]
            for k in range(1, _NUM_VARS + 1):
                vals = zone0.variables[k - 1].values
                assert vals is not None
                assert getattr(z1, f"var_{k}").dtype == vals.dtype

    # -- Options: --compress -----------------------------------------------------------

    def test_compress_reduces_size(self, tmp_path: Path) -> None:
        """-c yields a smaller file than the uncompressed conversion."""
        plain = tmp_path / "plain.mat"
        comp = tmp_path / "comp.mat"
        assert tec2mat(["-o", str(plain), str(_ONERA["szplt"])]) == 0
        assert tec2mat(["-c", "-o", str(comp), str(_ONERA["szplt"])]) == 0
        assert comp.stat().st_size < plain.stat().st_size

    def test_compress_roundtrips(self, tmp_path: Path) -> None:
        """A compressed file still loads with the expected structure."""
        dst = tmp_path / "comp.mat"
        assert tec2mat(["--compress", "-o", str(dst), str(_ONERA["szplt"])]) == 0
        info = _struct(_load_mat(dst), "info")
        assert int(info.num_zones) == _NUM_ZONES

    # -- Options: --oned-as ------------------------------------------------------------

    def test_oned_as_column_default(self, tmp_path: Path) -> None:
        """1-D FE variable arrays default to N x 1 column vectors."""
        dst = tmp_path / "out.mat"
        assert tec2mat(["-o", str(dst), str(_ONERA["szplt"])]) == 0
        z1 = _struct(_load_mat(dst, squeeze=False), "zone_1")
        assert z1.var_1.shape == (_Z1_NODES, 1)

    def test_oned_as_row(self, tmp_path: Path) -> None:
        """--oned-as row stores 1-D FE variable arrays as 1 x N row vectors."""
        dst = tmp_path / "out.mat"
        ret = tec2mat(["--oned-as", "row", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        z1 = _struct(_load_mat(dst, squeeze=False), "zone_1")
        assert z1.var_1.shape == (1, _Z1_NODES)

    # -- Passive / Shared / Active (synthetic IJK data) --------------------------------

    def test_synthetic_zone1_all_active(self, tmp_path: Path) -> None:
        """In the synthetic file, zone 1 has three active variables with data."""
        src = tmp_path / "syn.dat"
        _write_synthetic_ijk(src)
        dst = tmp_path / "syn.mat"
        assert tec2mat(["-o", str(dst), str(src)]) == 0
        z1 = _struct(_load_mat(dst), "zone_1")
        assert [str(s) for s in np.atleast_1d(z1.var_status)] == [
            "active",
            "active",
            "active",
        ]
        for k in (1, 2, 3):
            assert getattr(z1, f"var_{k}").size == 4

    def test_synthetic_passive_and_shared(self, tmp_path: Path) -> None:
        """Zone 2: var 1 shared, var 2 passive (both empty), var 3 active."""
        src = tmp_path / "syn.dat"
        _write_synthetic_ijk(src)
        dst = tmp_path / "syn.mat"
        assert tec2mat(["-o", str(dst), str(src)]) == 0
        z2 = _struct(_load_mat(dst), "zone_2")
        assert [str(s) for s in np.atleast_1d(z2.var_status)] == [
            "shared",
            "passive",
            "active",
        ]
        # Shared and passive variables carry no data (MATLAB []).
        assert np.size(z2.var_1) == 0
        assert np.size(z2.var_2) == 0
        # The active variable retains its data.
        assert np.size(z2.var_3) == 4
        # var 1 is shared from zone 1 (1-based); the others are not shared.
        assert np.array_equal(np.atleast_1d(z2.var_shared_from), [1, 0, 0])

    # -- Connectivity sharing ----------------------------------------------------------

    def test_synthetic_node_map_shared_from(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """A zone sharing FE connectivity stores node_map_shared_from, not a duplicate.

        The IJK-based synthetic fixture above can't exercise this at all --
        ordered zones have no connectivity -- so this reuses the
        ``shared_dataset``/``shared_path`` fixture (see TestSharingPreservation)
        instead, which is FE-based with real con_sharing.
        """
        dst = tmp_path / "shared.mat"
        assert tec2mat(["-o", str(dst), "--force", str(shared_path)]) == 0
        d = _load_mat(dst)
        z1 = _struct(d, "zone_1")
        z2 = _struct(d, "zone_2")
        assert hasattr(z1, "node_map")
        assert not hasattr(z2, "node_map")
        assert int(z2.node_map_shared_from) == 1

    # -- Error and overwrite paths -----------------------------------------------------

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """A non-existent source file returns exit code 1."""
        assert tec2mat([str(tmp_path / "ghost.dat")]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """An existing output without --force returns exit code 1."""
        dst = tmp_path / "out.mat"
        dst.touch()
        assert tec2mat(["-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "out.mat"
        dst.touch()
        ret = tec2mat(["-f", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert dst.stat().st_size > 0


# ======================================================================================
# tecaux
# ======================================================================================


class TestTecaux:
    """Tests for tecaux - adds dataset-, zone-, or variable-level auxiliary data."""

    # -- Dataset level aux -------------------------------------------------------------

    def test_dataset_aux_added(self, shared_path: Path, tmp_path: Path) -> None:
        """-d KEY=VALUE sets dataset-level auxiliary data."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-d",
            "Solver=MyCFD",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert dict(r.auxdata.items())["Solver"] == "MyCFD"

    def test_repeated_dataset_aux_flags_all_applied(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """Repeating -d accumulates multiple pairs, not just the last one."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-d",
            "Solver=MyCFD",
            "-d",
            "Version=2.1",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            aux = dict(r.auxdata.items())
            assert aux["Solver"] == "MyCFD"
            assert aux["Version"] == "2.1"

    def test_default_output_naming(self, shared_path: Path, tmp_path: Path) -> None:
        """With no -o, output is <stem>_aux<ext> next to the input."""
        src = tmp_path / f"flow{shared_path.suffix}"
        shutil.copy(shared_path, src)
        ret = tecaux(["-d", "Solver=MyCFD", str(src)])
        assert ret == 0
        assert (tmp_path / f"flow_aux{shared_path.suffix}").exists()

    # -- Zone level aux ----------------------------------------------------------------

    def test_zone_aux_specific_zone_only(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """-z INDEX KEY=VALUE applies only to that zone."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-z",
            "0",
            "Description=Wing",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert dict(r.zones[0].auxdata.items())["Description"] == "Wing"
            assert "Description" not in dict(r.zones[1].auxdata.items())

    def test_zone_aux_repeated_same_zone_merges(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """Multiple -z occurrences on the same zone merge and not overwrite."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-z",
            "0",
            "Description=Wing",
            "-z",
            "0",
            "Area=120sqm",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            aux = dict(r.zones[0].auxdata.items())
            assert aux["Description"] == "Wing"
            assert aux["Area"] == "120sqm"

    def test_zone_aux_all_broadcasts_to_every_zone(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """-z all KEY=VALUE applies to every zone."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-z",
            "all",
            "Batch=2024",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            for i in range(r.num_zones):
                assert dict(r.zones[i].auxdata.items())["Batch"] == "2024"

    def test_zone_aux_broadcast_and_specific_combine(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """A broadcast (-z all) and a specific (-z N) target compose in one call."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-z",
            "all",
            "Batch=2024",
            "-z",
            "0",
            "Special=true",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            aux1 = dict(r.zones[0].auxdata.items())
            aux2 = dict(r.zones[1].auxdata.items())
            assert aux1["Batch"] == "2024"
            assert aux1["Special"] == "true"
            assert aux2["Batch"] == "2024"
            assert "Special" not in aux2

    def test_zone_aux_out_of_range_returns_1(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """A zone index beyond num_zones returns exit code 1."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-z",
            "99",
            "X=1",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 1

    # -- Variable level aux ------------------------------------------------------------

    def test_var_aux_by_name(self, shared_path: Path, tmp_path: Path) -> None:
        """-v NAME KEY=VALUE resolves the variable case-insensitively by name."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-v",
            "c",
            "Units=Pa",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            # "c" is variable 3 (0-based) in the shared fixture.
            assert dict(r.get_var_auxdata(3).items())["Units"] == "Pa"

    def test_var_aux_by_index(self, shared_path: Path, tmp_path: Path) -> None:
        """-v INDEX KEY=VALUE resolves the variable by 0-based index."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-v",
            "0",
            "Units=m",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert dict(r.get_var_auxdata(0).items())["Units"] == "m"

    def test_var_aux_all_broadcasts_to_every_variable(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """-v all KEY=VALUE applies to every variable."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-v",
            "all",
            "Source=Test",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            for i in range(r.num_vars):
                assert dict(r.get_var_auxdata(i).items())["Source"] == "Test"

    def test_var_aux_unresolvable_returns_1(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """An unresolvable variable target (bad name and non-valid index) returns 1."""
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-v",
            "nonexistent",
            "X=1",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 1

    # -- Test preserve sharing ---------------------------------------------------------

    def test_everything_at_once_single_pass_preserves_sharing(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """Add all aux item types checking that sharing is preserved.

        This is the central thing tecaux has to get right: it's built on the same
        verbatim zone-copy approach as tecfix, so adding aux data must never require
        materializing a shared variable's or zone's data independently.

        """
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-d",
            "Solver=MyCFD",
            "-z",
            "0",
            "Case=A",
            "-z",
            "1",
            "Case=B",
            "-v",
            "w",
            "Units=K",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert dict(r.auxdata.items())["Solver"] == "MyCFD"
            assert dict(r.zones[0].auxdata.items())["Case"] == "A"
            assert dict(r.zones[1].auxdata.items())["Case"] == "B"
            assert dict(r.get_var_auxdata(4).items())["Units"] == "K"
            # The whole point: sharing relationships from the source file
            # are still exactly what they were.
            assert r.zones[1].variables[0].shared_zone == 0
            assert r.zones[2].shared_connectivity == 0
            np.testing.assert_allclose(
                r.zones[1].variables[0].values.ravel(), [0.0, 1.0, 0.0, 0.0]
            )

    def test_existing_aux_preserved_across_chained_runs(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """Running tecaux again on its own output keeps what an earlier run added."""
        first = tmp_path / "first.dat"
        second = tmp_path / "second.dat"
        assert (
            tecaux([
                "-d",
                "Solver=MyCFD",
                "-o",
                str(first),
                "--force",
                str(shared_path),
            ])
            == 0
        )
        assert (
            tecaux(["-d", "Version=2.1", "-o", str(second), "--force", str(first)]) == 0
        )
        with tecio.open(str(second), "r") as r:
            aux = dict(r.auxdata.items())
            assert aux["Solver"] == "MyCFD"
            assert aux["Version"] == "2.1"

    def test_output_format_controlled_by_extension(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """Writing to a different extension than the input converts format too."""
        dst = tmp_path / f"out{_other_format_flag(shared_path).replace('-', '.')}"
        ret = tecaux([
            "-d",
            "Solver=MyCFD",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        assert _is_readable(dst)

    # -- JSON input --------------------------------------------------------------------

    def test_json_all_three_levels_applied(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """A JSON file with AUXDATASET/AUXZONE/AUXVAR applies at all three levels."""
        json_path = tmp_path / "meta.json"
        json_path.write_text(
            json.dumps({
                "AUXDATASET": {"Solver": "FromJSON"},
                "AUXZONE": {"0": {"Description": "Wing"}},
                "AUXVAR": {"c": {"Units": "Pa"}},
            })
        )
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-j",
            str(json_path),
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert dict(r.auxdata.items())["Solver"] == "FromJSON"
            assert dict(r.zones[0].auxdata.items())["Description"] == "Wing"
            assert dict(r.get_var_auxdata(3).items())["Units"] == "Pa"

    def test_json_all_sentinel_broadcasts(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """The "all" key in AUXZONE/AUXVAR broadcasts, matching -z all/-v all."""
        json_path = tmp_path / "meta.json"
        json_path.write_text(json.dumps({"AUXZONE": {"all": {"Batch": "json-batch"}}}))
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-j",
            str(json_path),
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            for i in range(r.num_zones):
                assert dict(r.zones[i].auxdata.items())["Batch"] == "json-batch"

    def test_json_cli_override_wins_on_collision(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """A CLI -d flag overrides a JSON AUXDATASET value with the same key."""
        json_path = tmp_path / "meta.json"
        json_path.write_text(
            json.dumps({"AUXDATASET": {"Solver": "FromJSON", "Extra": "kept"}})
        )
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-j",
            str(json_path),
            "-d",
            "Solver=FromCLI",
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            aux = dict(r.auxdata.items())
            assert aux["Solver"] == "FromCLI"
            assert aux["Extra"] == "kept"

    def test_json_unrecognized_top_level_key_returns_1(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """A JSON file using the old/wrong key names (e.g. "dataset") is rejected."""
        json_path = tmp_path / "bad.json"
        json_path.write_text(json.dumps({"dataset": {"Solver": "MyCFD"}}))
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-j",
            str(json_path),
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 1

    def test_json_malformed_returns_1(self, shared_path: Path, tmp_path: Path) -> None:
        """Genuinely invalid JSON syntax fails cleanly, not with a raw traceback."""
        json_path = tmp_path / "bad.json"
        json_path.write_text("{not valid json")
        dst = tmp_path / "out.dat"
        ret = tecaux([
            "-j",
            str(json_path),
            "-o",
            str(dst),
            "--force",
            str(shared_path),
        ])
        assert ret == 1

    # -- Error and overwrite paths -----------------------------------------------------

    def test_malformed_kv_returns_1(self, shared_path: Path, tmp_path: Path) -> None:
        """A -d value without '=' returns exit code 1."""
        dst = tmp_path / "out.dat"
        ret = tecaux(["-d", "NoEquals", "-o", str(dst), "--force", str(shared_path)])
        assert ret == 1

    def test_no_flags_verbatim_copy_with_warning(
        self, shared_path: Path, tmp_path: Path, capsys
    ) -> None:
        """No -d/-z/-v/-j prints a warning to stderr and still writes a full copy."""
        dst = tmp_path / "out.dat"
        ret = tecaux(["-o", str(dst), "--force", str(shared_path)])
        assert ret == 0
        assert "no -d, -z, -v, or -j" in capsys.readouterr().err
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 3

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """A non-existent source file returns exit code 1."""
        assert tecaux(["-d", "X=1", str(tmp_path / "ghost.dat")]) == 1

    def test_existing_output_no_force_returns_1(
        self, shared_path: Path, tmp_path: Path
    ) -> None:
        """An existing output without --force returns exit code 1."""
        dst = tmp_path / "out.dat"
        dst.touch()
        ret = tecaux(["-d", "X=1", "-o", str(dst), str(shared_path)])
        assert ret == 1

    def test_force_overwrites_existing(self, shared_path: Path, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "out.dat"
        dst.touch()
        ret = tecaux(["-d", "X=1", "-f", "-o", str(dst), str(shared_path)])
        assert ret == 0
        assert dst.stat().st_size > 0

    # -- Strip / Export json --------------------------------------------------------

    def test_strip_and_export_together(self, aux_path: Path, tmp_path: Path) -> None:
        """--strip --export-json in one call covers the mechanical happy path for both.

        Uses ``aux_path`` (see ``_write_aux_dataset`` above), not
        ``shared_path`` -- the latter deliberately has no pre-existing aux
        data, so it can't exercise stripping/exporting anything real.
        """
        src = tmp_path / f"flow{aux_path.suffix}"
        shutil.copy(aux_path, src)
        strip_dst = tmp_path / f"stripped{aux_path.suffix}"

        ret = tecaux([
            "--strip",
            "--export-json",
            "-o",
            str(strip_dst),
            "--force",
            str(src),
        ])
        assert ret == 0

        # Stripped file: nothing left at any level, sharing still intact
        with tecio.open(str(strip_dst), "r") as r:
            assert dict(r.auxdata.items()) == {}
            assert dict(r.zones[0].auxdata.items()) == {}
            assert dict(r.zones[2].auxdata.items()) == {}
            assert dict(r.get_var_auxdata(0).items()) == {}
            assert dict(r.get_var_auxdata(3).items()) == {}
            assert r.zones[1].variables[0].shared_zone == 0
            assert r.zones[2].shared_connectivity == 0
            np.testing.assert_allclose(
                r.zones[1].variables[0].values.ravel(), [0.0, 1.0, 0.0, 0.0]
            )

        # JSON: exactly the original (pre-strip) data, sparse
        json_dst = src.with_name(f"{src.stem}_aux.json")
        assert json_dst.exists()
        with open(json_dst) as f:
            dumped = json.load(f)
        assert dumped["AUXDATASET"] == {"Solver": "MyCFD", "Version": "2.1"}
        assert dumped["AUXZONE"] == {
            "0": {"Description": "Wing"},
            "2": {"Description": "Empennage"},
        }
        assert dumped["AUXVAR"] == {"0": {"Units": "m"}, "3": {"Units": "Pa"}}

        # And it round-trips: feeding the export back in via -j reproduces the original
        # aux data on a fresh copy
        reimported = tmp_path / f"reimported{aux_path.suffix}"
        assert (
            tecaux(["-j", str(json_dst), "-o", str(reimported), "--force", str(src)])
            == 0
        )
        with tecio.open(str(reimported), "r") as r:
            assert dict(r.auxdata.items())["Solver"] == "MyCFD"
            assert dict(r.zones[0].auxdata.items())["Description"] == "Wing"

    def test_export_json_alone_leaves_source_untouched(
        self, aux_path: Path, tmp_path: Path
    ) -> None:
        """--export-json without --strip never modifies the source file.

        Only checkable without --strip: with --strip also given, the source is
        *expected* to be left alone too (only the separate strip output changes), so
        this specific guarantee has no combined-mode equivalent.
        """
        src = tmp_path / f"flow{aux_path.suffix}"
        shutil.copy(aux_path, src)
        before = src.read_bytes()

        assert tecaux(["--export-json", str(src)]) == 0

        assert src.read_bytes() == before
        assert (tmp_path / "flow_aux.json").exists()

    def test_strip_alone_does_not_write_json(
        self, aux_path: Path, tmp_path: Path
    ) -> None:
        """--strip without --export-json produces no JSON file at all.

        Regression guard: if a future edit ever dropped the ``if args.export_json:``
        guard around the JSON-writing step, a test that only ever exercised the combined
        case would never notice, since a JSON file would legitimately exist either way.
        """
        src = tmp_path / f"flow{aux_path.suffix}"
        shutil.copy(aux_path, src)

        assert tecaux(["--strip", str(src)]) == 0

        assert (tmp_path / f"flow_no_aux{aux_path.suffix}").exists()
        assert not (tmp_path / "flow_aux.json").exists()

    def test_output_flag_ignored_with_warning_for_export_only(
        self, aux_path: Path, tmp_path: Path, capsys
    ) -> None:
        """-o is meaningless (and warned about) for --export-json without --strip.

        Only reachable in export-only mode: when --strip is also given, -o legitimately
        applies to the strip output, so this warning path can't be exercised through the
        combined call at all.
        """
        src = tmp_path / f"flow{aux_path.suffix}"
        shutil.copy(aux_path, src)
        ignored_path = tmp_path / "ignored.dat"

        ret = tecaux(["--export-json", "-o", str(ignored_path), str(src)])
        assert ret == 0
        assert "is ignored" in capsys.readouterr().err
        assert not ignored_path.exists()
        assert (tmp_path / "flow_aux.json").exists()

    def test_strip_export_rejects_add_flags(
        self, aux_path: Path, tmp_path: Path
    ) -> None:
        """--strip/--export-json combined with -d/-z/-v/-j is rejected."""
        dst = tmp_path / f"out{aux_path.suffix}"
        assert tecaux(["--strip", "-d", "X=1", "-o", str(dst), str(aux_path)]) == 1
        assert tecaux(["--export-json", "-z", "1", "X=1", str(aux_path)]) == 1


# --------------------------------------------------------------------------------------
# Direct execution
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
