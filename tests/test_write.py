#!/usr/bin/env python3
"""pytest tests for :class:`tecio.{szl,plt,dat}.Write`.

All three ``Write`` classes are deliberately built to an identical public API
(``write_ordered_zone``, ``write_fe_zone``, aux data, sharing, ...); this suite runs the
*same* test body against all three formats, parametrized by ``fmt``, so a behavior only
has to be described once and is verified consistent across every writer.

Pattern per test:
    1. Create data with specific dtypes
    2. Write file via ``tecio.open(..., "w")`` — extension determines format
    3. Read back via ``tecio.open(..., "r")``
    4. Assert on metadata and values, using the ``_expected_dtype``/``_rtol`` helpers
       below wherever behavior legitimately differs by format

Format-specific dtype behavior
-------------------------------
Every writer provides a ``precision`` keyword but each resolve it differently.

* SZL (``precision: DataType | None = None``): with no override (the
  default), every dtype is preserved exactly (FLOAT, DOUBLE, INT32, INT16,
  BYTE) via automatic per-variable inference. An explicit ``precision`` overrides *only* variables
  whose own inferred type is FLOAT or DOUBLE; INT32/INT16/BYTE variables
  always keep their own type regardless, since SZL's new API has real
  per-variable type storage and there's no reason a precision setting
  should touch an integer.

* PLT (``precision: DataType = DataType.DOUBLE``, no automatic mode):
  confirmed empirically (a real pytest run against the C library, not just
  inferred from docs) that ``VIsDouble`` is a whole-file setting at
  ``tecini142`` time.

* DAT (``precision: DataType = DataType.FLOAT``, default single): the ASCII writer
  supports per-variable typing via the ``DT=`` zone-header keyword (Tecplot's own
  vocabulary: SINGLE/DOUBLE/LONGINT/ SHORTINT/BYTE), so ``precision`` only overrides
  floating-point-inferred variables; integers are always preserved exactly. The one
  thing to hold in mind is the *default* itself: ``DataType.FLOAT``, matching real
  Tecplot ASCII writers, means a ``float64`` input is downcast to SINGLE unless
  ``precision=DataType.DOUBLE`` is passed explicitly.

Tests that assert on ``variable.data_type`` use :func:`_expected_dtype` to get the
per-format, per-precision expected value.

Not parametrized here (kept as dedicated tests further down):

* ``DATAPACKING=POINT`` — real, working feature only in DAT; SZL/PLT reject it with
  ``NotImplementedError`` (verified once, parametrized over just those two).
* ``precision`` override semantics themselves — each writer has unique rules and
  defaults
* ``flush=`` incremental-flush kwarg on the SZL zone writers

Run directly:

    $ python tests/test_write.py -v

Keep output files for Tecplot 360 inspection:

    $ python tests/test_write.py -v --keep-files

"""

# ruff: noqa: E501, SIM117

import contextlib
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from create_test_data import (
    create_FE_brick,
    create_FE_lineseg,
    create_FE_prism,
    create_FE_pyramid,
    create_FE_quad,
    create_FE_tet,
    create_FE_tri,
    create_FE_two_bricks,
    create_ordered,
    create_poly_bconn_coarse,
    create_poly_bconn_fine,
    create_poly_golden_spiral,
    create_poly_hanging_ladder,
    create_poly_octagon,
    create_poly_two_cells,
    create_polyhedron_bconn_coarse,
    create_polyhedron_bconn_fine,
    create_polyhedron_cube,
    create_polyhedron_cube_pair,
    create_polyhedron_hanging_columns,
    create_polyhedron_mixed_faces,
    scalar_field,
)

import tecio
from tecio import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
    libtecio,
)
from tecio._szl_write import _infer_data_type

_RTOL_F32 = 1e-5
_RTOL_F64 = 1e-10
_RTOL_DAT = 1e-7

# ======================================================================================
# Cross-format helpers
# ======================================================================================

#: Every writer format under test, and the file extension tecio.open()
#: dispatches on to select it.
FORMATS: list[str] = ["szl", "plt", "dat"]
_EXTENSIONS: dict[str, str] = {"szl": "szplt", "plt": "plt", "dat": "dat"}


@pytest.fixture(params=FORMATS)
def fmt(request) -> str:
    """Parametrize a test across every writer format (szl, plt, dat)."""
    return request.param


def _path(output_path: Callable, fmt: str, name: str):
    """Build an output path with the extension for *fmt*."""
    return output_path(f"{name}.{_EXTENSIONS[fmt]}")


#: Each format's default precision
_DEFAULT_PRECISION: dict[str, DataType | None] = {
    "szl": None,
    "plt": DataType.DOUBLE,
    "dat": DataType.FLOAT,
}


def _expected_dtype(
    fmt: str, arr: np.ndarray, precision: DataType | str | None = None
) -> DataType:
    """Return the :class:`DataType` a variable of *arr*'s dtype reads back as.

    *precision* should mirror whatever (if anything) was passed to ``tecio.open(...,
    precision=...)`` in the test. ``None`` each format's own ``Write`` class default.

    * SZL: with no override, every dtype is preserved exactly. An explicit precision*
      *overrides only variables whose own inferred type is FLOAT or DOUBLE;
      *INT32/INT16/BYTE are always preserved regardless.
    * PLT: has no automatic mode at all -- *every* variable, float or integer, is
      written at the single resolved precision (default DOUBLE).
    * DAT: under its default (FLOAT), floating-inferred variables (float32 or* float64
      *input) are written as FLOAT; integer-inferred variables are always preserved
      *regardless of precision -- the same override rule as SZL, just a different
      *default.

    """
    if isinstance(precision, str):
        precision = (
            DataType.FLOAT
            if precision.strip().lower() in ("single", "float")
            else DataType.DOUBLE
        )
    resolved = precision if precision is not None else _DEFAULT_PRECISION[fmt]
    inferred = _infer_data_type(arr.dtype)

    if fmt == "plt":
        # Precision always resolves to a concrete DataType for PLT, and applies to every
        # variable
        assert resolved is not None
        return resolved

    # szl / dat: override applies only to floating-point-inferred variables
    if resolved is None:
        return inferred
    if inferred in (DataType.FLOAT, DataType.DOUBLE):
        return resolved
    return inferred


def _rtol(fmt: str, arr: np.ndarray) -> float:
    """Return the comparison tolerance appropriate for *fmt* and *arr*'s dtype.

    DAT's ASCII precision (default single, 9 significant digits) is the tightest bound.
    SZL/PLT round-trip through binary storage, so the array's own dtype sets the bound:
    float32 data only round-trips to float32 precision even once upcast to DOUBLE on
    disk (PLT).
    """
    if fmt == "dat":
        return _RTOL_DAT
    return _RTOL_F32 if arr.dtype == np.float32 else _RTOL_F64


def _rtol_for_precision(precision: DataType) -> float:
    """Return a tolerance tight enough to catch a precision override *not* actually
    taking effect.
    """
    return _RTOL_F32 if precision == DataType.FLOAT else 1e-13


# ======================================================================================
# Ordered (IJK) zone tests
# ======================================================================================


class TestWriteIJKZone:
    """Tests for write_ordered_zone, run against every writer format."""

    def test_write_ijk_3d_mixed_dtypes(self, fmt: str, output_path: Callable) -> None:
        """3-D ordered zone with mixed float32/float64 variables.

        Demonstrates:
        - Basic ``write_ordered_zone`` call structure: ``data``, ``variables``, ``title``
        - Zone dimensions ``(imax, jmax, kmax)`` inferred from the first nodal
          array's shape
        - Per-format dtype behavior (see module docstring)
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        z = z.astype(np.float32)
        c = scalar_field(x, y, z).astype(np.float64)

        path = _path(output_path, fmt, "write_ijk_3d")
        with tecio.open(str(path), "w") as w:
            w.write_ordered_zone(
                data=[x, y, z, c], variables=["x", "y", "z", "c"], title="zone_3d"
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 1
            assert r.num_vars == 4
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.ORDERED
            assert zone.dimensions == (i, j, k)
            assert zone.variables[0].data_type == _expected_dtype(fmt, x)
            assert zone.variables[1].data_type == _expected_dtype(fmt, y)
            np.testing.assert_allclose(
                zone.variables[0].values.ravel(), x.ravel(), rtol=_rtol(fmt, x)
            )
            np.testing.assert_allclose(
                zone.variables[1].values.ravel(), y.ravel(), rtol=_rtol(fmt, y)
            )

    def test_write_ijk_cell_centered(self, fmt: str, output_path: Callable) -> None:
        """3-D zone with nodal coordinates and a cell-centered scalar.

        Demonstrates:
        - ``value_locations=[NODAL, NODAL, NODAL, CELL_CENTERED]``
        - Cell-centered array shape ``(imax-1, jmax-1, kmax-1)``, validated and inferred
          identically across formats
        """
        i, j, k = 4, 5, 3
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        cc = np.random.default_rng(42).random((i - 1, j - 1, k - 1)).astype(np.float64)

        path = _path(output_path, fmt, "write_ijk_cc")
        with tecio.open(str(path), "w") as w:
            w.write_ordered_zone(
                data=[x, y, z, cc],
                variables=["x", "y", "z", "cc"],
                value_locations=[
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.CELL_CENTERED,
                ],
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            cc_var = r.zones[0].variables[3]
            assert cc_var.value_location == ValueLocation.CELL_CENTERED
            assert cc_var.data_type == _expected_dtype(fmt, cc)
            assert cc_var.values.size == (i - 1) * (j - 1) * (k - 1)
            np.testing.assert_allclose(
                cc_var.values.ravel(), cc.ravel(), rtol=_rtol(fmt, cc)
            )

    def test_write_ijk_int_variable(self, fmt: str, output_path: Callable) -> None:
        """Ordered zone with INT32/INT16/BYTE variables.

        Demonstrates:
        - Integer casts: ``(field * scale).astype(np.intXX)`` / ``.astype(np.uint8)``
        - SZL and DAT both preserve each integer type exactly; PLT upcasts everything
          (including integers) to DOUBLE, since ``tecdat142`` has no native integer
          storage
        """
        n = 10
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c_i32 = (np.sin(2 * np.pi * x) * 1000).astype(np.int32)
        c_i16 = (np.sin(2 * np.pi * x) * 100).astype(np.int16)
        c_u8 = ((np.sin(2 * np.pi * x) + 1.0) * 127).astype(np.uint8)

        path = _path(output_path, fmt, "write_ijk_int")
        with tecio.open(str(path), "w") as w:
            w.write_ordered_zone(
                data=[x, c_i32, c_i16, c_u8],
                variables=["x", "c_i32", "c_i16", "c_u8"],
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.variables[1].data_type == _expected_dtype(fmt, c_i32)
            assert zone.variables[2].data_type == _expected_dtype(fmt, c_i16)
            assert zone.variables[3].data_type == _expected_dtype(fmt, c_u8)
            np.testing.assert_allclose(
                zone.variables[1].values.ravel(), c_i32.astype(np.float64)
            )
            np.testing.assert_allclose(
                zone.variables[2].values.ravel(), c_i16.astype(np.float64)
            )
            np.testing.assert_allclose(
                zone.variables[3].values.ravel(), c_u8.astype(np.float64)
            )

    def test_write_ijk_unsteady(self, fmt: str, output_path: Callable) -> None:
        """Transient dataset with variable sharing for grid coordinates.

        Demonstrates:
        - ``strand_id``/``solution_time`` for animation; zones sharing a strand_id
          animate together in Tecplot 360
        - ``var_sharing={"x": 0, "y": 0, "z": 0}``: x, y, z shared from zone 0 by
          name; only the scalar is supplied for later zones
        """
        i, j, k = 6, 5, 4
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 20)

        path = _path(output_path, fmt, "write_ijk_unsteady")
        with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as w:
            for t in solution_times:
                c = scalar_field(x + t, y + t, z).astype(np.float64)
                w.write_ordered_zone(
                    data=[x, y, z, c] if w.current_zone == -1 else [c],
                    var_sharing=(
                        None if w.current_zone == -1 else {"x": 0, "y": 0, "z": 0}
                    ),
                    strand_id=1,
                    solution_time=float(t),
                )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == len(solution_times)
            assert r.zones[0].solution_time == pytest.approx(0.0)
            assert r.zones[-1].solution_time == pytest.approx(solution_times[-1])
            assert r.zones[5].strand_id == 1
            assert r.zones[1].variables[0].shared_zone is not None

    def test_write_ijk_shared_var_dimensions_from_source(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Zone dimensions are taken from the shared source, not the local array.

        Demonstrates:
        - A second zone shares x, y, z via ``var_sharing={"x": 0, "y": 0, "z": 0}``
          and supplies only the scalar; ``data`` contains only active variables
        - The writer resolves ``(imax, jmax, kmax)`` from the shared source
        - Read-back: shared coordinate reports ``shared_zone``; local scalar reads its
          own value and resolves through as if local
        """
        i, j, k = 4, 5, 3
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c0 = scalar_field(x, y, z).astype(np.float64)
        c1 = (scalar_field(x, y, z) * 2.0).astype(np.float64)

        path = _path(output_path, fmt, "write_ijk_shared_dims")
        with tecio.open(str(path), "w") as w:
            w.write_ordered_zone(
                data=[x, y, z, c0], variables=["x", "y", "z", "c"], title="zone_1"
            )
            w.write_ordered_zone(
                data=[c1], var_sharing={"x": 0, "y": 0, "z": 0}, title="zone_2"
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            assert r.zones[1].dimensions == (i, j, k)
            assert r.zones[1].variables[0].shared_zone is not None
            # Shared variable forwards the source zone's data, exactly as if
            # it were local.
            np.testing.assert_allclose(
                r.zones[1].variables[0].values.ravel(), x.ravel(), rtol=_rtol(fmt, x)
            )
            assert r.zones[1].variables[3].shared_zone is None
            np.testing.assert_allclose(
                r.zones[1].variables[3].values.ravel(), c1.ravel(), rtol=_rtol(fmt, c1)
            )

    def test_write_ijk_passive_variable(self, fmt: str, output_path: Callable) -> None:
        """Passive variable in an ordered zone.

        Demonstrates:
        - ``passive_vars={"unused"}``; ``data`` supplies only the active arrays
        """
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)

        path = _path(output_path, fmt, "write_ijk_passive")
        with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as w:
            w.write_ordered_zone(data=[x, c], passive_vars={"unused"})

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zones[0].variables[1].is_passive()

    def test_write_ijk_passive_variable_by_index(
        self, fmt: str, output_path: Callable
    ) -> None:
        """``passive_vars`` also accepts 0-based variable indices."""
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)

        path = _path(output_path, fmt, "write_ijk_passive_by_index")
        with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as w:
            w.write_ordered_zone(data=[x, c], passive_vars={1})

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zones[0].variables[1].is_passive()

    def test_write_ijk_passive_variables_set(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Marking more than one variable passive."""
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)

        path = _path(output_path, fmt, "write_ijk_passive_set")
        with tecio.open(
            str(path), "w", variables=["x", "unused1", "unused2", "c"]
        ) as w:
            w.write_ordered_zone(data=[x, c], passive_vars={"unused1", "unused2"})

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zones[0].variables[1].is_passive()
            assert r.zones[0].variables[2].is_passive()

    def test_write_ijk_var_sharing_by_index(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Sparse ``var_sharing`` also accepts 0-based variable indices as keys."""
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c0 = np.sin(2 * np.pi * x).astype(np.float64)
        c1 = (np.sin(2 * np.pi * x) * 2.0).astype(np.float64)

        path = _path(output_path, fmt, "write_ijk_sharing_by_index")
        with tecio.open(str(path), "w", variables=["x", "c"]) as w:
            w.write_ordered_zone(data=[x, c0], title="zone_1")
            w.write_ordered_zone(data=[c1], var_sharing={0: 0}, title="zone_2")

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zones[1].variables[0].shared_zone is not None

    def test_write_ijk_var_sharing_mixed_name_and_index_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A ``var_sharing`` mapping mixing name and index keys raises TypeError.

        Without this, a name and an index key referring to the same variable could
        set conflicting share targets, with one silently overwriting the other.
        """
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c0 = np.sin(2 * np.pi * x).astype(np.float64)
        c1 = (np.sin(2 * np.pi * x) * 2.0).astype(np.float64)

        path = _path(output_path, fmt, "write_ijk_sharing_mixed_raises")
        with tecio.open(str(path), "w", variables=["x", "c"]) as w:
            w.write_ordered_zone(data=[x, c0], title="zone_1")
            with pytest.raises(TypeError):
                w.write_ordered_zone(
                    data=[c1], var_sharing={"x": 0, 0: 0}, title="zone_2"
                )

    def test_write_ijk_passive_and_shared_conflict_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A variable named in both sparse passive_vars and var_sharing raises."""
        i, j, k = 3, 3, 1
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c0 = scalar_field(x, y, z).astype(np.float64)
        path = _path(output_path, fmt, "write_ijk_passive_shared_conflict")

        with pytest.raises(ValueError, match="passive"):
            with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as w:
                w.write_ordered_zone(data=[x, y, z, c0], title="zone_1")
                w.write_ordered_zone(
                    data=[],
                    passive_vars={"x"},
                    var_sharing={"x": 0, "y": 0, "z": 0},
                    title="zone_2",
                )

    def test_write_ijk_sparse_unknown_name_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """An unrecognised variable name in sparse passive_vars raises KeyError."""
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        path = _path(output_path, fmt, "write_ijk_sparse_unknown_name")

        with pytest.raises(KeyError):
            with tecio.open(str(path), "w", variables=["x", "c"]) as w:
                w.write_ordered_zone(data=[x], passive_vars={"nonexistent"})

    def test_write_ijk_sparse_index_out_of_range_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A 0-based index outside the variable list in passive_vars raises."""
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        path = _path(output_path, fmt, "write_ijk_sparse_index_out_of_range")

        with pytest.raises(IndexError):
            with tecio.open(str(path), "w", variables=["x", "c"]) as w:
                w.write_ordered_zone(data=[x], passive_vars={5})

    def test_write_ijk_passive_vars_bare_value_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A bare name/index (not in a list/tuple/set) raises TypeError.

        In particular a bare string must be rejected outright rather than
        silently iterated character-by-character.
        """
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        path = _path(output_path, fmt, "write_ijk_passive_vars_bare_value")

        with pytest.raises(TypeError):
            with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as w:
                w.write_ordered_zone(data=[x], passive_vars="unused")

        with pytest.raises(TypeError):
            with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as w:
                w.write_ordered_zone(data=[x], passive_vars=2)

    def test_write_ijk_all_shared_warns_and_skips(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A zone with no active variables warns and is skipped, not an error.

        Demonstrates:
        - Every variable shared -> :class:`tecio.EmptyZoneWarning`, no
          :exc:`ValueError`
        - The writer is still usable afterward: a following zone writes
          normally, and the skipped zone never appears in the file
        """
        i, j, k = 3, 3, 1
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c0 = scalar_field(x, y, z).astype(np.float64)
        c1 = (scalar_field(x, y, z) * 2.0).astype(np.float64)
        path = _path(output_path, fmt, "write_ijk_all_shared_warns")

        with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as w:
            w.write_ordered_zone(data=[x, y, z, c0], title="zone_1")
            with pytest.warns(tecio.EmptyZoneWarning):
                w.write_ordered_zone(
                    data=[],
                    var_sharing={"x": 0, "y": 0, "z": 0, "c": 0},
                    title="zone_skipped",
                )
            assert w.current_zone == 0  # skipped zone did not advance the count
            w.write_ordered_zone(data=[x, y, z, c1], title="zone_2")

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            assert [zn.title for zn in r.zones] == ["zone_1", "zone_2"]

    def test_write_ijk_dataset_and_zone_aux(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Dataset-level and zone-level auxiliary data.

        Demonstrates:
        - ``add_auxdataset_dict`` buffers dataset-level metadata, flushed before the
          first zone
        - ``aux={...}`` on ``write_ordered_zone`` attaches zone-level metadata
        """
        x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        dataset_aux = {"Solver": "TestCode", "Mach": "0.72"}
        zone_aux = {"MeshType": "structured", "Author": "pytest"}

        path = _path(output_path, fmt, "write_ijk_aux")
        with tecio.open(str(path), "w") as w:
            w.add_auxdataset_dict(dataset_aux)
            w.write_ordered_zone(data=[x], variables=["x"], aux=zone_aux)

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            for k, v in dataset_aux.items():
                assert r.auxdata[k] == v
            for k, v in zone_aux.items():
                assert r.zones[0].auxdata[k] == v

    def test_write_ijk_file_type_grid(self, fmt: str, output_path: Callable) -> None:
        """``FileType.GRID`` survives the round-trip for every format."""
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        path = _path(output_path, fmt, "write_ijk_grid")

        with tecio.open(str(path), "w", file_type=FileType.GRID) as w:
            w.write_ordered_zone(data=[x], variables=["x"])

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.file_type == FileType.GRID

    # ----------------------------------------------------------------------------------
    # Error paths
    # ----------------------------------------------------------------------------------

    def test_write_ijk_var_count_mismatch_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Fewer data arrays than active variables raises ValueError."""
        x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        y = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        path = _path(output_path, fmt, "write_ijk_var_count_mismatch")

        with pytest.raises(ValueError, match="[Ee]xpected"):
            with tecio.open(str(path), "w", variables=["x", "y", "c"]) as w:
                w.write_ordered_zone(data=[x, y])  # missing c

    def test_write_ijk_shape_mismatch_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Inconsistent nodal array shapes raise ValueError."""
        i, j, k = 4, 5, 1
        x, y, _ = create_ordered((i, j, k))
        x = x.squeeze(-1)
        y_bad = y.squeeze(-1)[:-1, :]
        path = _path(output_path, fmt, "write_ijk_shape_mismatch")

        with pytest.raises(ValueError), tecio.open(str(path), "w") as w:
            w.write_ordered_zone(data=[x, y_bad], variables=["x", "y"])

    def test_write_ijk_shared_var_shape_mismatch_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A shared variable's shape is validated against its true source, not itself.

        Demonstrates:
        - The array supplied for the local variable is squeezed to 2-D, which is
          internally self-consistent -- validation that only compared locally-supplied
          arrays would miss this
        - The writer validates against the shared source's actual recorded dimensions
          instead, catching the mismatch
        """
        i, j, k = 4, 5, 3
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = scalar_field(x, y, z).astype(np.float64)
        bad_c = c[:, :, 0]  # drops k entirely
        path = _path(output_path, fmt, "write_ijk_shared_shape_mismatch")

        with pytest.raises(ValueError):
            with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as w:
                w.write_ordered_zone(data=[x, y, z, c])
                w.write_ordered_zone(data=[bad_c], var_sharing={"x": 0, "y": 0, "z": 0})


# ======================================================================================
# Finite-element zone tests
# ======================================================================================


class TestWriteFEZone:
    """Tests for write_fe_zone, run against every writer format."""

    def test_write_fe_lineseg(self, fmt: str, output_path: Callable) -> None:
        """FELINESEG -- two-node line segment elements."""
        x, y, nodes = create_FE_lineseg()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)

        path = _path(output_path, fmt, "write_fe_lineseg")
        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FELINESEG,
                data=[x, y, c],
                node_map=nodes,
                variables=["x", "y", "c"],
                title="FE_LineSeg",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FELINESEG
            assert zone.num_nodes == len(x)
            assert zone.num_elements == len(nodes)
            np.testing.assert_allclose(zone.variables[0].values, x, rtol=_rtol(fmt, x))
            np.testing.assert_array_equal(zone.node_map, nodes.astype(np.int64))

    def test_write_fe_tri(self, fmt: str, output_path: Callable) -> None:
        """FETRIANGLE -- three-node triangular elements, mixed dtypes."""
        x, y, nodes = create_FE_tri()
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        c = (scalar_field(x, y) * 1000).astype(np.int32)

        path = _path(output_path, fmt, "write_fe_tri")
        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, y, c],
                node_map=nodes,
                variables=["x", "y", "c"],
                title="FE_Tri",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FETRIANGLE
            assert zone.num_nodes == 4
            assert zone.num_elements == 2
            assert zone.variables[0].data_type == _expected_dtype(fmt, x)
            assert zone.variables[2].data_type == _expected_dtype(fmt, c)
            np.testing.assert_allclose(zone.variables[0].values, x, rtol=_rtol(fmt, x))
            np.testing.assert_allclose(zone.variables[1].values, y, rtol=_rtol(fmt, y))
            np.testing.assert_allclose(zone.variables[2].values, c.astype(np.float64))

    def test_write_fe_quad(self, fmt: str, output_path: Callable) -> None:
        """FEQUADRILATERAL -- four-node quadrilateral elements."""
        x, y, nodes = create_FE_quad()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        c = scalar_field(x, y).astype(np.float32)

        path = _path(output_path, fmt, "write_fe_quad")
        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FEQUADRILATERAL,
                data=[x, y, c],
                node_map=nodes,
                variables=["x", "y", "c"],
                title="FE_Quad",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FEQUADRILATERAL
            assert zone.num_nodes == 6
            assert zone.num_elements == 2
            assert zone.variables[0].data_type == _expected_dtype(fmt, x)
            np.testing.assert_allclose(zone.variables[0].values, x, rtol=_rtol(fmt, x))

    def test_write_fe_tet(self, fmt: str, output_path: Callable) -> None:
        """FETETRAHEDRON -- four-node tetrahedral elements (3-D volume mesh)."""
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = scalar_field(x, y).astype(np.float64)

        path = _path(output_path, fmt, "write_fe_tet")
        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FETETRAHEDRON,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Tet",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FETETRAHEDRON
            assert zone.num_nodes == 5
            assert zone.num_elements == 2
            assert zone.variables[3].data_type == _expected_dtype(fmt, c)
            np.testing.assert_array_equal(zone.node_map, nodes.astype(np.int64))

    def test_write_fe_pyramid(self, fmt: str, output_path: Callable) -> None:
        """Pyramid as a degenerate FEBRICK (nodes 5-8 collapsed to the apex)."""
        x, y, z, nodes = create_FE_pyramid()
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        z = z.astype(np.float32)
        c = (scalar_field(x, y) * 100).astype(np.int16)

        path = _path(output_path, fmt, "write_fe_pyramid")
        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Pyramid",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FEBRICK
            assert zone.variables[3].data_type == _expected_dtype(fmt, c)
            np.testing.assert_allclose(zone.variables[3].values, c.astype(np.float64))

    def test_write_fe_prism(self, fmt: str, output_path: Callable) -> None:
        """Triangular prism as a degenerate FEBRICK (repeated edge nodes)."""
        x, y, z, nodes = create_FE_prism()
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        z = z.astype(np.float64)
        c = ((scalar_field(x, y) + 1.0) * 127).astype(np.uint8)

        path = _path(output_path, fmt, "write_fe_prism")
        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Prism",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.variables[3].data_type == _expected_dtype(fmt, c)
            np.testing.assert_allclose(zone.variables[3].values, c.astype(np.float64))

    def test_write_fe_brick(self, fmt: str, output_path: Callable) -> None:
        """FEBRICK -- standard 8-node hexahedral elements."""
        x, y, z, _faces, nodes = create_FE_brick()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float64)
        c = (scalar_field(x, y) * 1000).astype(np.int32)

        path = _path(output_path, fmt, "write_fe_brick")
        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Brick",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FEBRICK
            assert zone.num_nodes == 8
            assert zone.num_elements == 1
            assert zone.variables[2].data_type == _expected_dtype(fmt, z)
            np.testing.assert_allclose(zone.variables[2].values, z, rtol=_rtol(fmt, z))
            np.testing.assert_allclose(zone.variables[3].values, c.astype(np.float64))

    def test_write_fe_face_neighbors(self, fmt: str, output_path: Callable) -> None:
        """Two FEBRICK cells with face-neighbor connectivity and a CC variable.

        SZL specifically raises NotImplementedError instead of writing (likely a bug in
        the TecIO C library)
        """
        x, y, z, nodes, face_neighbors = create_FE_two_bricks()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = np.array([1.1, 2.2], dtype=np.float64)  # one value per element

        path = _path(output_path, fmt, "write_fe_face_neighbors")
        kwargs = {
            "zone_type": ZoneType.FEBRICK,
            "data": [x, y, z, c],
            "node_map": nodes,
            "variables": ["x", "y", "z", "c"],
            "title": "FE_2Bricks",
            "value_locations": [
                ValueLocation.NODAL,
                ValueLocation.NODAL,
                ValueLocation.NODAL,
                ValueLocation.CELL_CENTERED,
            ],
            "face_neighbors": face_neighbors,
            # face_neighbor_mode omitted deliberately: exercises the new
            # "face_neighbors given without a mode defaults to
            # LOCAL_ONE_TO_ONE" behavior, not just the explicit path.
        }

        if fmt == "szl":
            w = tecio.open(str(path), "w")
            try:
                with pytest.raises(NotImplementedError, match="face-neighbor"):
                    w.write_fe_zone(**kwargs)
            finally:
                with contextlib.suppress(Exception):
                    w.close()
            return

        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(**kwargs)

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.num_elements == 2
            cc_var = zone.variables[3]
            assert cc_var.value_location == ValueLocation.CELL_CENTERED
            np.testing.assert_allclose(cc_var.values, c, rtol=_rtol(fmt, c))

            # The face-neighbor data itself, not just an unrelated variable,
            # this is the round trip that actually matters for this test.
            assert zone.face_neighbor_mode == FaceNeighborMode.LOCAL_ONE_TO_ONE
            assert zone.num_face_connections == len(face_neighbors)
            np.testing.assert_array_equal(
                zone.get_face_connections(reshape=True), face_neighbors
            )

    def test_write_fe_face_neighbor_mode_without_data_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """face_neighbor_mode given without face_neighbors is a mistake, raises."""
        x, y, nodes = create_FE_tri()

        path = _path(output_path, fmt, "write_fe_face_neighbor_mode_only")
        w = tecio.open(str(path), "w")
        try:
            with pytest.raises(ValueError, match="face_neighbor_mode"):
                w.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x, y],
                    node_map=nodes,
                    variables=["x", "y"],
                    face_neighbor_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
                )
        finally:
            # No zone was ever successfully written; closing is expected to
            # be a no-op or fail here, this test only cares that the
            # validation itself raised, not that a valid file results.
            with contextlib.suppress(Exception):
                w.close()

    def test_write_fe_face_neighbors_shared_connectivity(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Sharing connectivity bypasses face-neighbor data entirely.

        Providing face_neighbors alongside con_sharing doesn't raise (a
        local mode is a valid combination), but the data is never actually
        written, it's implicitly inherited from the source zone, matching
        how node_map itself is skipped when connectivity is shared.

        SZL is broken and raises NotImplementedError on the very first write_fe_zone.
        """
        x, y, z, nodes, face_neighbors = create_FE_two_bricks()

        path = _path(output_path, fmt, "write_fe_face_neighbors_shared")

        if fmt == "szl":
            w = tecio.open(str(path), "w")
            try:
                with pytest.raises(NotImplementedError, match="face-neighbor"):
                    w.write_fe_zone(
                        zone_type=ZoneType.FEBRICK,
                        data=[x, y, z],
                        node_map=nodes,
                        variables=["x", "y", "z"],
                        title="source",
                        face_neighbors=face_neighbors,
                    )
            finally:
                with contextlib.suppress(Exception):
                    w.close()
            return

        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z],
                node_map=nodes,
                variables=["x", "y", "z"],
                title="source",
                face_neighbors=face_neighbors,
            )
            w.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z],
                variables=["x", "y", "z"],
                title="shared",
                con_sharing=0,
                face_neighbors=face_neighbors,
                face_neighbor_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
            )
            source_meta = w.meta.zone(0)
            shared_meta = w.meta.zone(1)

        assert source_meta.face_neighbor_mode == FaceNeighborMode.LOCAL_ONE_TO_ONE
        assert source_meta.num_face_connections == len(face_neighbors)
        assert shared_meta.face_neighbor_mode is None
        assert shared_meta.num_face_connections is None

        with tecio.open(str(path), "r") as r:
            source_zone = r.zones[0]
            shared_zone = r.zones[1]
            assert source_zone.num_face_connections == len(face_neighbors)
            np.testing.assert_array_equal(
                source_zone.get_face_connections(reshape=True), face_neighbors
            )
            assert shared_zone.num_face_connections is None

    def test_write_fe_face_neighbors_global_mode_with_sharing_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A global face-neighbor mode can't be combined with con_sharing.

        Per the classic API's own documented constraint: connectivity (and
        any face-neighbor data with it) can only be shared for local modes.
        """
        x, y, z, nodes, face_neighbors = create_FE_two_bricks()

        path = _path(output_path, fmt, "write_fe_face_neighbors_global_shared")
        w = tecio.open(str(path), "w")
        try:
            w.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z],
                node_map=nodes,
                variables=["x", "y", "z"],
                title="source",
            )
            with pytest.raises(ValueError, match="global"):
                w.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z],
                    variables=["x", "y", "z"],
                    title="shared",
                    con_sharing=0,
                    face_neighbors=face_neighbors,
                    face_neighbor_mode=FaceNeighborMode.GLOBAL_ONE_TO_ONE,
                )
        finally:
            with contextlib.suppress(Exception):
                w.close()

    @pytest.mark.parametrize(
        ("bad_face_neighbors", "match"),
        [
            (np.array([[0, 99, 1]]), "face 99"),
            (np.array([[99, 1, 1]]), "cell 99"),
        ],
        ids=["bad_face_index", "bad_cell_index"],
    )
    def test_write_fe_face_neighbors_out_of_range_raises(
        self,
        fmt: str,
        output_path: Callable,
        bad_face_neighbors: np.ndarray,
        match: str,
    ) -> None:
        """Structurally invalid cell/face references raise, not silently written."""
        x, y, z, nodes, _ = create_FE_two_bricks()

        path = _path(output_path, fmt, "write_fe_face_neighbors_bad_index")
        w = tecio.open(str(path), "w")
        try:
            with pytest.raises(ValueError, match=match):
                w.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z],
                    node_map=nodes,
                    variables=["x", "y", "z"],
                    face_neighbors=bad_face_neighbors,
                )
        finally:
            with contextlib.suppress(Exception):
                w.close()

    def test_write_fe_face_neighbors_degenerate_cell(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A cell with a repeated node doesn't trip up validation.

        E.g. a triangle written as a quad, with its last node repeated.
        The degenerate face (between the repeated node and itself) has no
        possible neighbor, and real Tecplot-generated data omits it
        entirely, confirmed against a real exported file (a sliced,
        triangle-only region of Onera.szplt, converted to degenerate quads
        and re-exported with GUI-generated face neighbors). Validation only
        bounds-checks against the cell type's face count, it never requires
        an exact per-cell count, so it doesn't reject this.

        SZL still raises NotImplementedError (see test_write_fe_face_neighbors
        for why), but only *after* validation itself passes the degenerate
        case cleanly, confirming the degeneracy handling isn't what's
        broken, it's specifically the known SZL library limitation.
        """
        x, y, nodes = create_FE_quad()
        # Degenerate cell 0: repeat its last node, mimicking a triangle
        # written as a quad, exactly how the reference file above was
        # constructed (repeat the last node_map column).
        nodes = nodes.copy()
        nodes[0, 3] = nodes[0, 2]  # cell 0: [0, 1, 4, 3] -> [0, 1, 4, 4]

        # Both connections reference only real (non-degenerate) faces:
        # cell 0's face 1 and cell 1's face 3, the shared edge 1-4. Neither
        # references face 2 (now degenerate), matching what Tecplot itself
        # omits for a cell like this.
        face_neighbors = np.array([[0, 1, 1], [1, 3, 0]], dtype=np.int64)

        path = _path(output_path, fmt, "write_fe_face_neighbors_degenerate")

        if fmt == "szl":
            w = tecio.open(str(path), "w")
            try:
                with pytest.raises(NotImplementedError, match="face-neighbor"):
                    w.write_fe_zone(
                        zone_type=ZoneType.FEQUADRILATERAL,
                        data=[x, y],
                        node_map=nodes,
                        variables=["x", "y"],
                        face_neighbors=face_neighbors,
                    )
            finally:
                with contextlib.suppress(Exception):
                    w.close()
            return

        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FEQUADRILATERAL,
                data=[x, y],
                node_map=nodes,
                variables=["x", "y"],
                face_neighbors=face_neighbors,
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.num_face_connections == 2
            np.testing.assert_array_equal(
                zone.get_face_connections(reshape=True), face_neighbors
            )

    def test_write_fe_passive_variable(self, fmt: str, output_path: Callable) -> None:
        """Passive variable in an FE zone -- identical behavior to ordered zones."""
        x, y, nodes = create_FE_tri()
        x = x.astype(np.float32)
        c = scalar_field(x, y).astype(np.float64)

        path = _path(output_path, fmt, "write_fe_passive")
        with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as w:
            w.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, c],
                node_map=nodes,
                passive_vars={"unused"},
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zones[0].variables[1].is_passive()

    def test_write_fe_unsteady(self, fmt: str, output_path: Callable) -> None:
        """FETETRAHEDRON zones with strand ID, solution time, and sharing.

        Demonstrates:
        - FE zones cannot share coordinates across zones without also sharing
          connectivity (node maps differ per-zone in general), so this exercises both
          ``var_sharing`` and ``con_sharing`` together
        - ``var_sharing={"x": 0, "y": 0, "z": 0}``: sparse, by name, same as the
          ordered-zone case
        - ``shared_connectivity`` and shared-variable forwarding are
          consistent across all three readers
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 20)

        path = _path(output_path, fmt, "write_fe_unsteady")
        with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as w:
            for step, t in enumerate(solution_times):
                c = np.sin(x + t).astype(np.float64)
                w.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c] if w.current_zone == -1 else [c],
                    var_sharing=(
                        None if w.current_zone == -1 else {"x": 0, "y": 0, "z": 0}
                    ),
                    node_map=nodes if w.current_zone == -1 else None,
                    con_sharing=None if w.current_zone == -1 else 0,
                    title=f"zone_t{step + 1}",
                    strand_id=1,
                    solution_time=float(t),
                    aux={"MeshType": "unstructured", "Author": "test_write"},
                )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == len(solution_times)
            assert r.zones[0].solution_time == pytest.approx(0.0)
            assert r.zones[-1].solution_time == pytest.approx(solution_times[-1])
            assert r.zones[0].strand_id == 1
            assert r.zones[1].shared_connectivity is not None
            np.testing.assert_array_equal(r.zones[1].node_map, nodes.astype(np.int64))
            assert r.zones[1].variables[0].shared_zone is not None
            np.testing.assert_allclose(
                r.zones[1].variables[0].values, x, rtol=_rtol(fmt, x)
            )

    # ----------------------------------------------------------------------------------
    # Error paths
    # ----------------------------------------------------------------------------------

    def test_write_fe_var_count_mismatch_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Too few data arrays for active variables raises ValueError."""
        x, y, nodes = create_FE_tri()
        path = _path(output_path, fmt, "write_fe_var_count_mismatch")

        with pytest.raises(ValueError, match="[Ee]xpected"):
            with tecio.open(str(path), "w", variables=["x", "y", "c"]) as w:
                w.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE, data=[x, y], node_map=nodes
                )

    def test_write_fe_array_length_mismatch_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Nodal array shorter than num_nodes raises ValueError."""
        x, y, nodes = create_FE_tri()
        path = _path(output_path, fmt, "write_fe_array_length_mismatch")

        with pytest.raises(ValueError), tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x[:-1], y],
                node_map=nodes,
                variables=["x", "y"],
            )

    def test_write_fe_shared_var_length_mismatch_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A shared variable's implied length is validated, not just node_map.

        Demonstrates:
        - With connectivity and coordinates shared from zone 0 (``con_sharing=0``,
          ``var_sharing={"x": 0, "y": 0, "z": 0}``), an incorrectly-sized scalar is
          still caught even though ``node_map`` itself is omitted for this zone
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = np.sin(x).astype(np.float64)
        path = _path(output_path, fmt, "write_fe_shared_var_length_mismatch")

        with pytest.raises(ValueError, match="value"):
            with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as w:
                w.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                )
                w.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[c[0:-2]],
                    var_sharing={"x": 0, "y": 0, "z": 0},
                    con_sharing=0,
                )

    def test_write_fe_unsupported_zone_type_raises(
        self, fmt: str, output_path: Callable
    ) -> None:
        """FEPOLYGON is face-based: every writer points to ``write_poly_zone``."""
        x, y, nodes = create_FE_tri()
        path = _path(output_path, fmt, "write_fe_unsupported_zone_type")

        with pytest.raises(ValueError, match="write_poly_zone"):
            with tecio.open(str(path), "w") as w:
                w.write_fe_zone(
                    zone_type=ZoneType.FEPOLYGON,
                    data=[x, y],
                    node_map=nodes,
                    variables=["x", "y"],
                )

    def test_write_fe_all_shared_warns_and_skips(
        self, fmt: str, output_path: Callable
    ) -> None:
        """An FE zone with no active variables warns and is skipped.

        Demonstrates:
        - Coordinates and connectivity both shared from zone 0
          (``con_sharing=0``, ``var_sharing`` covering every variable)
          -> :class:`tecio.EmptyZoneWarning`, no :exc:`ValueError`
        - The writer is still usable afterward
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c0 = np.sin(x).astype(np.float64)
        c1 = np.cos(x).astype(np.float64)
        path = _path(output_path, fmt, "write_fe_all_shared_warns")

        with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as w:
            w.write_fe_zone(
                zone_type=ZoneType.FETETRAHEDRON,
                data=[x, y, z, c0],
                node_map=nodes,
                title="zone_1",
            )
            with pytest.warns(tecio.EmptyZoneWarning):
                w.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[],
                    var_sharing={"x": 0, "y": 0, "z": 0, "c": 0},
                    con_sharing=0,
                    title="zone_skipped",
                )
            assert w.current_zone == 0  # skipped zone did not advance the count
            w.write_fe_zone(
                zone_type=ZoneType.FETETRAHEDRON,
                data=[x, y, z, c1],
                node_map=nodes,
                title="zone_2",
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            assert [zn.title for zn in r.zones] == ["zone_1", "zone_2"]


# ======================================================================================
# Poly (FEPOLYGON / FEPOLYHEDRON) zone tests (PLT and DAT only)
# ======================================================================================

#: Formats that can hold poly zones.
POLY_FORMATS: list[str] = ["plt", "dat"]


@pytest.mark.parametrize("fmt", POLY_FORMATS)
class TestWritePolyZone:
    """Tests for write_poly_zone, run against the formats that support it."""

    def test_write_poly_octagon(self, fmt: str, output_path: Callable) -> None:
        """FEPOLYGON -- a single eight-sided cell.

        Demonstrates:
        - ``write_poly_zone`` call structure: ``zone_type``, ``data``, and the face map
          ``face_nodes``/``face_left_elements``/``face_right_elements``
        - A polygon face is an edge with 2 nodes, so no ``face_node_counts`` is given
        - 0-based indices, with ``-1`` for no element on that side of a face
        """
        x, y, faces, left, right = create_poly_octagon()

        path = _path(output_path, fmt, "write_poly_octagon")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                variables=["x", "y"],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
                title="Poly_Octagon",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert isinstance(zone, tecio.TecplotPolyZoneReader)
            assert zone.title == "Poly_Octagon"
            assert zone.zone_type == ZoneType.FEPOLYGON
            assert (zone.num_nodes, zone.num_elements, zone.num_faces) == (8, 1, 8)
            np.testing.assert_array_equal(zone.face_nodes, faces.ravel())
            np.testing.assert_array_equal(zone.face_left_elements, left)
            np.testing.assert_array_equal(zone.face_right_elements, right)
            np.testing.assert_allclose(zone.variables[0].values, x, rtol=_rtol(fmt, x))

    def test_write_poly_two_cells(self, fmt: str, output_path: Callable) -> None:
        """FEPOLYGON -- a pentagon and a triangle sharing one edge."""
        x, y, faces, left, right = create_poly_two_cells()

        path = _path(output_path, fmt, "write_poly_two_cells")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert (zone.num_nodes, zone.num_elements, zone.num_faces) == (6, 2, 7)
            np.testing.assert_array_equal(zone.face_nodes, faces.ravel())
            np.testing.assert_array_equal(zone.face_left_elements, left)
            np.testing.assert_array_equal(zone.face_right_elements, right)
            # The reader reports 2 nodes for every polygon face.
            np.testing.assert_array_equal(zone.face_node_counts, 2)

    def test_write_poly_hanging_ladder(self, fmt: str, output_path: Callable) -> None:
        """FEPOLYGON -- columns of 1, 2 and 4 cells with hanging nodes.

        The coordinates are integers, which are kept as integers in a DAT file.
        """
        x, y, faces, left, right = create_poly_hanging_ladder()

        path = _path(output_path, fmt, "write_poly_hanging_ladder")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert (zone.num_nodes, zone.num_elements, zone.num_faces) == (15, 7, 21)
            np.testing.assert_array_equal(zone.face_nodes, faces.ravel())
            np.testing.assert_array_equal(zone.face_left_elements, left)
            np.testing.assert_array_equal(zone.face_right_elements, right)
            np.testing.assert_array_equal(zone.variables[0].values, x)

    def test_write_poly_golden_spiral(self, fmt: str, output_path: Callable) -> None:
        """FEPOLYGON -- five squares laid out as a golden spiral."""
        x, y, faces, left, right = create_poly_golden_spiral()

        path = _path(output_path, fmt, "write_poly_golden_spiral")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert (zone.num_nodes, zone.num_elements, zone.num_faces) == (12, 5, 16)
            np.testing.assert_array_equal(zone.face_nodes, faces.ravel())
            np.testing.assert_array_equal(zone.face_left_elements, left)
            np.testing.assert_array_equal(zone.face_right_elements, right)

    def test_write_poly_bconn(self, fmt: str, output_path: Callable) -> None:
        """Two FEPOLYGON zones joined by connected boundary faces.

        Demonstrates:
        - ``boundary_connection_counts``/``_elements``/``_zones`` on the second zone,
          which points at elements of zone 0 (TecIO only accepts connections to a zone
          that is already written)
        - ``face_right_elements`` values ``-2`` and ``-3`` reference boundary faces 0 and 1
        """
        fx, fy, ffaces, fleft, fright = create_poly_bconn_fine()
        cx, cy, cfaces, cleft, cright, counts, elems, zones = create_poly_bconn_coarse()

        path = _path(output_path, fmt, "write_poly_bconn")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[fx, fy],
                face_nodes=ffaces,
                face_left_elements=fleft,
                face_right_elements=fright,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[cx, cy],
                face_nodes=cfaces,
                face_left_elements=cleft,
                face_right_elements=cright,
                boundary_connection_counts=counts,
                boundary_connection_elements=elems,
                boundary_connection_zones=zones,
            )

        with tecio.open(str(path), "r") as r:
            fine, coarse = r.zones
            assert (fine.num_nodes, fine.num_elements, fine.num_faces) == (10, 4, 13)
            assert fine.num_boundary_faces == 0
            assert (coarse.num_nodes, coarse.num_elements, coarse.num_faces) == (
                6,
                2,
                7,
            )
            assert coarse.num_boundary_faces == 2
            assert coarse.num_boundary_connections == 4
            np.testing.assert_array_equal(coarse.face_right_elements, cright)
            np.testing.assert_array_equal(coarse.boundary_connection_counts, counts)
            np.testing.assert_array_equal(coarse.boundary_connection_elements, elems)
            np.testing.assert_array_equal(coarse.boundary_connection_zones, zones)

    def test_write_polyhedron_cube(self, fmt: str, output_path: Callable) -> None:
        """FEPOLYHEDRON -- one cube.

        Demonstrates:
        - ``face_node_counts`` is required for polyhedra, whose faces differ in size
        - Flat ``face_nodes`` that the reader splits with ``face_node_offsets``
        """
        x, y, z, nodes, counts, left, right = create_polyhedron_cube()

        path = _path(output_path, fmt, "write_polyhedron_cube")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[x, y, z],
                variables=["x", "y", "z"],
                face_nodes=nodes,
                face_node_counts=counts,
                face_left_elements=left,
                face_right_elements=right,
                title="Poly_Cube",
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FEPOLYHEDRON
            assert (zone.num_nodes, zone.num_elements, zone.num_faces) == (8, 1, 6)
            np.testing.assert_array_equal(zone.face_nodes, nodes)
            np.testing.assert_array_equal(zone.face_node_counts, counts)
            np.testing.assert_array_equal(zone.face_node_offsets, 4 * np.arange(7))
            np.testing.assert_array_equal(zone.face_left_elements, left)
            np.testing.assert_array_equal(zone.face_right_elements, right)
            np.testing.assert_array_equal(zone.variables[2].values, z)

    def test_write_polyhedron_cube_pair(self, fmt: str, output_path: Callable) -> None:
        """FEPOLYHEDRON -- two cubes sharing a face."""
        x, y, z, nodes, counts, left, right = create_polyhedron_cube_pair()

        path = _path(output_path, fmt, "write_polyhedron_cube_pair")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[x, y, z],
                face_nodes=nodes,
                face_node_counts=counts,
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert (zone.num_nodes, zone.num_elements, zone.num_faces) == (12, 2, 11)
            np.testing.assert_array_equal(zone.face_nodes, nodes)
            np.testing.assert_array_equal(zone.face_node_counts, counts)
            np.testing.assert_array_equal(zone.face_left_elements, left)
            np.testing.assert_array_equal(zone.face_right_elements, right)

    def test_write_polyhedron_mixed_faces(
        self, fmt: str, output_path: Callable
    ) -> None:
        """FEPOLYHEDRON -- cube, pyramid, prism and hexagonal prism in one zone."""
        x, y, z, nodes, counts, left, right = create_polyhedron_mixed_faces()

        path = _path(output_path, fmt, "write_polyhedron_mixed_faces")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[x, y, z],
                face_nodes=nodes,
                face_node_counts=counts,
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert (zone.num_nodes, zone.num_elements, zone.num_faces) == (23, 4, 22)
            assert set(zone.face_node_counts.tolist()) == {3, 4, 6}
            np.testing.assert_array_equal(zone.face_nodes, nodes)
            np.testing.assert_array_equal(zone.face_node_counts, counts)
            np.testing.assert_array_equal(zone.face_left_elements, left)
            np.testing.assert_array_equal(zone.face_right_elements, right)
            np.testing.assert_allclose(zone.variables[0].values, x, rtol=_rtol(fmt, x))

    def test_write_polyhedron_hanging_columns(
        self, fmt: str, output_path: Callable
    ) -> None:
        """FEPOLYHEDRON -- one coarse cube against four fine cubes."""
        x, y, z, nodes, counts, left, right = create_polyhedron_hanging_columns()

        path = _path(output_path, fmt, "write_polyhedron_hanging_columns")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[x, y, z],
                face_nodes=nodes,
                face_node_counts=counts,
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert (zone.num_nodes, zone.num_elements, zone.num_faces) == (22, 5, 25)
            np.testing.assert_array_equal(zone.face_nodes, nodes)
            np.testing.assert_array_equal(zone.face_node_counts, counts)
            np.testing.assert_array_equal(zone.face_left_elements, left)
            np.testing.assert_array_equal(zone.face_right_elements, right)

    def test_write_polyhedron_bconn(self, fmt: str, output_path: Callable) -> None:
        """Two FEPOLYHEDRON zones joined by one boundary face with four connections."""
        fx, fy, fz, fnodes, fcounts, fleft, fright = create_polyhedron_bconn_fine()
        cx, cy, cz, cnodes, ccounts, cleft, cright, bcounts, belems, bzones = (
            create_polyhedron_bconn_coarse()
        )

        path = _path(output_path, fmt, "write_polyhedron_bconn")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[fx, fy, fz],
                face_nodes=fnodes,
                face_node_counts=fcounts,
                face_left_elements=fleft,
                face_right_elements=fright,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[cx, cy, cz],
                face_nodes=cnodes,
                face_node_counts=ccounts,
                face_left_elements=cleft,
                face_right_elements=cright,
                boundary_connection_counts=bcounts,
                boundary_connection_elements=belems,
                boundary_connection_zones=bzones,
            )

        with tecio.open(str(path), "r") as r:
            fine, coarse = r.zones
            assert (fine.num_nodes, fine.num_elements, fine.num_faces) == (18, 4, 20)
            assert (coarse.num_nodes, coarse.num_elements, coarse.num_faces) == (
                8,
                1,
                6,
            )
            assert coarse.num_boundary_faces == 1
            assert coarse.num_boundary_connections == 4
            np.testing.assert_array_equal(coarse.face_right_elements, cright)
            np.testing.assert_array_equal(coarse.boundary_connection_counts, bcounts)
            np.testing.assert_array_equal(coarse.boundary_connection_elements, belems)
            np.testing.assert_array_equal(coarse.boundary_connection_zones, bzones)

    def test_write_poly_cell_centered(self, fmt: str, output_path: Callable) -> None:
        """A cell-centered variable on a poly zone has one value per element.

        Demonstrates:
        - ``value_locations=[NODAL, NODAL, CELL_CENTERED]``
        - Cell-centered array length ``num_elements``, validated like other zone types
        """
        x, y, faces, left, right = create_poly_two_cells()
        c = np.array([10.0, 20.0])

        path = _path(output_path, fmt, "write_poly_cell_centered")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y, c],
                variables=["x", "y", "c"],
                value_locations=[
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.CELL_CENTERED,
                ],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.variables["c"].value_location == ValueLocation.CELL_CENTERED
            np.testing.assert_allclose(zone.get_array("c"), c)

    def test_write_poly_default_title(self, fmt: str, output_path: Callable) -> None:
        """Without ``title`` the zone is named ``Poly_Zone_<index>``."""
        x, y, faces, left, right = create_poly_octagon()

        path = _path(output_path, fmt, "write_poly_default_title")
        with tecio.open(str(path), "w") as w:
            for _ in range(2):
                w.write_poly_zone(
                    zone_type=ZoneType.FEPOLYGON,
                    data=[x, y],
                    face_nodes=faces,
                    face_left_elements=left,
                    face_right_elements=right,
                )

        with tecio.open(str(path), "r") as r:
            assert [z.title for z in r.zones] == ["Poly_Zone_0", "Poly_Zone_1"]

    def test_write_polygon_counts_accepted_when_two(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A polygon's ``face_node_counts`` may be passed if every count is 2.

        This lets the arrays of a reader be passed straight back to the writer.
        """
        x, y, faces, left, right = create_poly_octagon()

        path = _path(output_path, fmt, "write_poly_counts_two")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_node_counts=np.full(8, 2),
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            np.testing.assert_array_equal(r.zones[0].face_nodes, faces.ravel())

    def test_write_poly_transient_and_aux(
        self, fmt: str, output_path: Callable
    ) -> None:
        """``solution_time``, ``strand_id`` and zone ``aux`` on a poly zone."""
        x, y, faces, left, right = create_poly_two_cells()

        path = _path(output_path, fmt, "write_poly_transient")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
                solution_time=0.5,
                strand_id=1,
                aux={"Key": "value"},
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.solution_time == 0.5
            assert zone.strand_id == 1
            assert dict(zone.auxdata.items()) == {"Key": "value"}

    def test_write_poly_with_other_zone_types(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A poly zone and an FE zone in one file keep their own reader classes."""
        fx, fy, nodes = create_FE_tri()
        x, y, faces, left, right = create_poly_two_cells()

        path = _path(output_path, fmt, "write_poly_with_fe")
        with tecio.open(str(path), "w") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[fx, fy],
                node_map=nodes,
                variables=["x", "y"],
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

        with tecio.open(str(path), "r") as r:
            assert isinstance(r.zones[0], tecio.TecplotFEZoneReader)
            assert isinstance(r.zones[1], tecio.TecplotPolyZoneReader)
            np.testing.assert_array_equal(r.zones[0].node_map, nodes.astype(np.int64))
            np.testing.assert_array_equal(r.zones[1].face_left_elements, left)


@pytest.mark.parametrize("fmt", POLY_FORMATS)
class TestWritePolySharing:
    """Connectivity, variable and passive handling for poly zones."""

    def test_write_poly_connectivity_and_variable_sharing(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Zone 1 shares x, y, z and the whole face map with zone 0.

        Demonstrates:
        - ``con_sharing=0`` writes no face map for the zone, and cannot be combined
          with face arrays
        - ``var_sharing`` and ``passive_vars`` work as in the other zone types
        - The reader returns the source zone's face map for the sharing zone
        """
        x, y, z, nodes, counts, left, right = create_polyhedron_cube_pair()
        f = x + 2 * y

        path = _path(output_path, fmt, "write_poly_shared")
        with tecio.open(str(path), "w", variables=["x", "y", "z", "f", "q"]) as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[x, y, z, f],
                face_nodes=nodes,
                face_node_counts=counts,
                face_left_elements=left,
                face_right_elements=right,
                passive_vars={"q"},
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[f + 1],
                var_sharing={"x": 0, "y": 0, "z": 0},
                passive_vars={"q"},
                con_sharing=0,
            )

        with tecio.open(str(path), "r") as r:
            first, second = r.zones
            assert first.shared_connectivity is None
            assert second.shared_connectivity == 0
            assert second.num_faces == first.num_faces == 11
            np.testing.assert_array_equal(second.face_nodes, nodes)
            np.testing.assert_array_equal(second.face_node_counts, counts)
            np.testing.assert_array_equal(second.face_left_elements, left)
            np.testing.assert_array_equal(second.face_right_elements, right)
            assert first.variables["x"].shared_zone is None
            assert second.variables["x"].shared_zone == 0
            assert second.variables["q"].is_passive()
            assert second.get_array("q") is None
            np.testing.assert_array_equal(second.get_array("x"), first.get_array("x"))
            np.testing.assert_allclose(second.get_array("f"), f + 1, rtol=_rtol(fmt, f))


@pytest.mark.parametrize("fmt", POLY_FORMATS)
class TestWritePolyAppend:
    """Poly zones survive the copy done by the append modes."""

    def test_append_keeps_and_adds_poly_zones(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Mode ``"a"`` keeps the zones already on disk and adds a new one."""
        ox, oy, ofaces, oleft, oright = create_poly_octagon()
        tx, ty, tfaces, tleft, tright = create_poly_two_cells()

        path = _path(output_path, fmt, "write_poly_append")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[ox, oy],
                variables=["x", "y"],
                face_nodes=ofaces,
                face_left_elements=oleft,
                face_right_elements=oright,
            )
        with tecio.open(str(path), "a") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[tx, ty],
                face_nodes=tfaces,
                face_left_elements=tleft,
                face_right_elements=tright,
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            np.testing.assert_array_equal(r.zones[0].face_nodes, ofaces.ravel())
            np.testing.assert_array_equal(r.zones[1].face_nodes, tfaces.ravel())
            np.testing.assert_array_equal(r.zones[1].face_right_elements, tright)

    def test_append_read_write_reads_existing_zone(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Mode ``"a+"`` reads the poly zones that are already in the file."""
        x, y, z, nodes, counts, left, right = create_polyhedron_cube()

        path = _path(output_path, fmt, "write_poly_append_rw")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[x, y, z],
                face_nodes=nodes,
                face_node_counts=counts,
                face_left_elements=left,
                face_right_elements=right,
            )
        with tecio.open(str(path), "a+") as rw:
            zone = rw.zones[0]
            assert isinstance(zone, tecio.TecplotPolyZoneReader)
            np.testing.assert_array_equal(zone.face_nodes, nodes)

    def test_append_keeps_boundary_connections_and_sharing(
        self, fmt: str, output_path: Callable
    ) -> None:
        """Re-writing a file keeps boundary connections and shared connectivity."""
        fx, fy, ffaces, fleft, fright = create_poly_bconn_fine()
        cx, cy, cfaces, cleft, cright, counts, elems, zones = create_poly_bconn_coarse()
        path = _path(output_path, fmt, "write_poly_append_bconn")
        with tecio.open(str(path), "w", variables=["x", "y"]) as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[fx, fy],
                face_nodes=ffaces,
                face_left_elements=fleft,
                face_right_elements=fright,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[cx, cy],
                face_nodes=cfaces,
                face_left_elements=cleft,
                face_right_elements=cright,
                boundary_connection_counts=counts,
                boundary_connection_elements=elems,
                boundary_connection_zones=zones,
            )
        shared = _path(output_path, fmt, "write_poly_append_shared")
        with tecio.open(str(shared), "w", variables=["x", "y"]) as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[fx, fy],
                face_nodes=ffaces,
                face_left_elements=fleft,
                face_right_elements=fright,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[fx + 1],
                var_sharing={"y": 0},
                con_sharing=0,
            )

        for p in (path, shared):
            with tecio.open(str(p), "a"):
                pass

        with tecio.open(str(path), "r") as r:
            np.testing.assert_array_equal(r.zones[1].boundary_connection_counts, counts)
            np.testing.assert_array_equal(
                r.zones[1].boundary_connection_elements, elems
            )
            np.testing.assert_array_equal(r.zones[1].boundary_connection_zones, zones)
        with tecio.open(str(shared), "r") as r:
            assert r.zones[1].shared_connectivity == 0
            assert r.zones[1].variables["y"].shared_zone == 0


@pytest.mark.parametrize("fmt", POLY_FORMATS)
class TestWritePolyErrors:
    """Inputs that cannot describe a valid face map are rejected before writing."""

    @pytest.fixture
    def path(self, fmt: str, output_path: Callable) -> Path:
        """Output file for a write that is expected to fail."""
        return _path(output_path, fmt, "write_poly_error")

    def test_fe_zone_with_poly_type_points_to_write_poly_zone(self, path: Path) -> None:
        """``write_fe_zone`` refuses the face-based zone types."""
        x, y, nodes = create_FE_tri()
        for zone_type in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            with (
                pytest.raises(ValueError, match="write_poly_zone"),
                tecio.open(str(path), "w") as w,
            ):
                w.write_fe_zone(zone_type=zone_type, data=[x, y], node_map=nodes)

    def test_poly_zone_with_fe_type_points_to_write_fe_zone(self, path: Path) -> None:
        """``write_poly_zone`` refuses the element-based zone types."""
        x, y, faces, left, right = create_poly_octagon()
        with (
            pytest.raises(ValueError, match="write_fe_zone"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

    def test_missing_face_arrays(self, path: Path) -> None:
        """Every zone that is not sharing connectivity needs the three face arrays."""
        x, y, faces, left, right = create_poly_octagon()
        for omit in ("face_nodes", "face_left_elements", "face_right_elements"):
            arrays = {
                "face_nodes": faces,
                "face_left_elements": left,
                "face_right_elements": right,
            }
            del arrays[omit]
            with (
                pytest.raises(ValueError, match="required"),
                tecio.open(str(path), "w") as w,
            ):
                w.write_poly_zone(zone_type=ZoneType.FEPOLYGON, data=[x, y], **arrays)

    def test_polyhedron_needs_counts(self, path: Path) -> None:
        """Polyhedron faces differ in size, so ``face_node_counts`` is required."""
        x, y, z, nodes, counts, left, right = create_polyhedron_cube()
        with (
            pytest.raises(ValueError, match="face_node_counts is required"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[x, y, z],
                face_nodes=nodes,
                face_left_elements=left,
                face_right_elements=right,
            )

    def test_polygon_counts_must_be_two(self, path: Path) -> None:
        """A polygon face is an edge, so a count other than 2 is an error."""
        x, y, faces, left, right = create_poly_octagon()
        with (
            pytest.raises(ValueError, match="always have 2 nodes"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_node_counts=np.full(8, 3),
                face_left_elements=left,
                face_right_elements=right,
            )

    def test_size_mismatches(self, path: Path) -> None:
        """Face arrays must agree with each other in length."""
        x, y, z, nodes, counts, left, right = create_polyhedron_cube()
        arrays = {
            "face_nodes": nodes,
            "face_node_counts": counts,
            "face_left_elements": left,
            "face_right_elements": right,
        }
        for key in ("face_right_elements", "face_node_counts", "face_nodes"):
            short = {**arrays, key: arrays[key][:-1]}
            with (
                pytest.raises(ValueError, match=f"{key} has"),
                tecio.open(str(path), "w") as w,
            ):
                w.write_poly_zone(
                    zone_type=ZoneType.FEPOLYHEDRON, data=[x, y, z], **short
                )

    def test_negative_node_index(self, path: Path) -> None:
        """Node indices are 0-based, so a negative index is an error."""
        x, y, faces, left, right = create_poly_octagon()
        faces = faces.copy()
        faces[0, 0] = -1
        with (
            pytest.raises(ValueError, match="0-based"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

    def test_non_integer_indices(self, path: Path) -> None:
        """Face arrays must be integers."""
        x, y, faces, left, right = create_poly_octagon()
        with (
            pytest.raises(ValueError, match="integer"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces.astype(float),
                face_left_elements=left,
                face_right_elements=right,
            )

    def test_no_element_referenced(self, path: Path) -> None:
        """At least one face must reference an element."""
        x, y, faces, left, right = create_poly_octagon()
        none = np.full(8, -1)
        with (
            pytest.raises(ValueError, match="No face references an element"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=none,
                face_right_elements=none,
            )

    def test_boundary_reference_without_boundary_faces(self, path: Path) -> None:
        """A ``-2`` in the face arrays needs boundary connection arrays."""
        x, y, faces, left, right = create_poly_octagon()
        right = right.copy()
        right[0] = -2
        with (
            pytest.raises(ValueError, match="boundary face 0"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

    def test_boundary_arrays_must_come_together(self, path: Path) -> None:
        """The three boundary connection arrays are all given or all omitted."""
        x, y, faces, left, right, counts, elems, zones = create_poly_bconn_coarse()
        with (
            pytest.raises(ValueError, match="given together"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
                boundary_connection_counts=counts,
                boundary_connection_elements=elems,
            )

    def test_boundary_count_mismatch(self, path: Path) -> None:
        """The boundary connection arrays must agree with each other."""
        x, y, faces, left, right, counts, elems, zones = create_poly_bconn_coarse()
        cases = [
            ("sums to", np.array([1, 1]), elems, zones),
            ("boundary_connection_zones has", counts, elems, np.array([0, 0])),
            ("positive", np.array([4, 0]), elems, zones),
        ]
        for message, bcounts, belems, bzones in cases:
            with (
                pytest.raises(ValueError, match=message),
                tecio.open(str(path), "w") as w,
            ):
                w.write_poly_zone(
                    zone_type=ZoneType.FEPOLYGON,
                    data=[x, y],
                    face_nodes=faces,
                    face_left_elements=left,
                    face_right_elements=right,
                    boundary_connection_counts=bcounts,
                    boundary_connection_elements=belems,
                    boundary_connection_zones=bzones,
                )

    def test_boundary_zone_must_exist(self, path: Path) -> None:
        """The Tecplot libraries do not check the zone index, so it is checked here."""
        x, y, faces, left, right, counts, elems, zones = create_poly_bconn_coarse()
        for bad_zone, message in (
            (5, "reference a zone already written"),
            (-1, "0-based"),
        ):
            bzones = zones.copy()
            bzones[0] = bad_zone
            with (
                pytest.raises(ValueError, match=message),
                tecio.open(str(path), "w") as w,
            ):
                w.write_poly_zone(
                    zone_type=ZoneType.FEPOLYGON,
                    data=[x, y],
                    face_nodes=faces,
                    face_left_elements=left,
                    face_right_elements=right,
                    boundary_connection_counts=counts,
                    boundary_connection_elements=elems,
                    boundary_connection_zones=bzones,
                )

    def test_boundary_element_must_exist_in_its_zone(self, path: Path) -> None:
        """The first zone written has no earlier zone, only itself to point at."""
        x, y, faces, left, right, counts, elems, zones = create_poly_bconn_coarse()
        with (
            pytest.raises(ValueError, match="past the"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
                boundary_connection_counts=counts,
                boundary_connection_elements=elems,
                boundary_connection_zones=np.zeros_like(zones),  # this zone itself
            )

    def test_boundary_element_checked_against_earlier_zone(self, path: Path) -> None:
        """A connection to an element past the end of zone 0 is an error."""
        fx, fy, ffaces, fleft, fright = create_poly_bconn_fine()
        x, y, faces, left, right, counts, elems, zones = create_poly_bconn_coarse()
        elems = elems.copy()
        elems[0] = 4  # one past the end of zone 0, which has 4 elements
        with (
            pytest.raises(ValueError, match="past the 4 elements of zone 0"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[fx, fy],
                face_nodes=ffaces,
                face_left_elements=fleft,
                face_right_elements=fright,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
                boundary_connection_counts=counts,
                boundary_connection_elements=elems,
                boundary_connection_zones=zones,
            )

    def test_con_sharing_rejects_face_arrays(self, path: Path) -> None:
        """A zone that shares connectivity cannot also give face arrays."""
        x, y, faces, left, right = create_poly_octagon()
        with (
            pytest.raises(ValueError, match="cannot be given together"),
            tecio.open(str(path), "w") as w,
        ):
            for con_sharing in (None, 0):
                w.write_poly_zone(
                    zone_type=ZoneType.FEPOLYGON,
                    data=[x, y],
                    face_nodes=faces,
                    face_left_elements=left,
                    face_right_elements=right,
                    con_sharing=con_sharing,
                )

    def test_con_sharing_needs_matching_poly_zone(self, path: Path) -> None:
        """The source zone must exist, be a poly zone, and have the same type."""
        fx, fy, nodes = create_FE_tri()
        x, y, faces, left, right = create_poly_octagon()

        with (
            pytest.raises(ValueError, match="not a poly zone"),
            tecio.open(str(path), "w", variables=["x", "y"]) as w,
        ):
            w.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE, data=[fx, fy], node_map=nodes
            )
            w.write_poly_zone(zone_type=ZoneType.FEPOLYGON, data=[x, y], con_sharing=0)

        with (
            pytest.raises(ValueError, match="cannot share connectivity"),
            tecio.open(str(path), "w", variables=["x", "y"]) as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON, data=[x, y], con_sharing=0
            )

        with (
            pytest.raises(ValueError, match="not been written"),
            tecio.open(str(path), "w", variables=["x", "y"]) as w,
        ):
            w.write_poly_zone(zone_type=ZoneType.FEPOLYGON, data=[x, y], con_sharing=3)

    def test_con_sharing_rejects_zone_with_boundary_faces(self, path: Path) -> None:
        """TecIO refuses to share a face map that has boundary faces."""
        fx, fy, ffaces, fleft, fright = create_poly_bconn_fine()
        x, y, faces, left, right, counts, elems, zones = create_poly_bconn_coarse()
        with (
            pytest.raises(ValueError, match="boundary faces"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[fx, fy],
                face_nodes=ffaces,
                face_left_elements=fleft,
                face_right_elements=fright,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
                boundary_connection_counts=counts,
                boundary_connection_elements=elems,
                boundary_connection_zones=zones,
            )
            w.write_poly_zone(zone_type=ZoneType.FEPOLYGON, data=[x, y], con_sharing=1)

    def test_variable_array_sizes_are_checked(self, path: Path) -> None:
        """Nodal arrays have one value per node, cell-centered one per element."""
        x, y, faces, left, right = create_poly_two_cells()
        locations = [
            ValueLocation.NODAL,
            ValueLocation.NODAL,
            ValueLocation.CELL_CENTERED,
        ]
        cases = [
            ("CELL_CENTERED", [x, y, np.zeros(1)]),  # 2 elements
            ("NODAL", [x[:-1], y, np.zeros(2)]),  # 6 nodes
        ]
        for message, data in cases:
            with (
                pytest.raises(ValueError, match=message),
                tecio.open(str(path), "w") as w,
            ):
                w.write_poly_zone(
                    zone_type=ZoneType.FEPOLYGON,
                    data=data,
                    value_locations=locations,
                    face_nodes=faces,
                    face_left_elements=left,
                    face_right_elements=right,
                )

    def test_all_variables_shared_warns_and_skips(
        self, fmt: str, output_path: Callable
    ) -> None:
        """A zone whose variables are all shared holds nothing and is skipped."""
        x, y, faces, left, right = create_poly_two_cells()

        path = _path(output_path, fmt, "write_poly_all_shared")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                variables=["x", "y"],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )
            with pytest.warns(tecio.EmptyZoneWarning):
                w.write_poly_zone(
                    zone_type=ZoneType.FEPOLYGON,
                    data=[],
                    con_sharing=0,
                    var_sharing={"x": 0, "y": 0},
                )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 1

    def test_point_packing_rejected(self, path: Path) -> None:
        """Poly zones only support BLOCK packing."""
        x, y, faces, left, right = create_poly_octagon()
        with (
            pytest.raises(ValueError, match="BLOCK data packing"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
                datapacking="POINT",
            )

    def test_bad_datapacking_string(self, path: Path) -> None:
        """An unknown ``datapacking`` string is an error."""
        x, y, faces, left, right = create_poly_octagon()
        with (
            pytest.raises(ValueError, match="datapacking"),
            tecio.open(str(path), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
                datapacking="X",
            )


class TestWritePolySzl:
    """SZPLT cannot hold polygon or polyhedron zones yet."""

    def test_write_poly_zone_not_implemented(self, output_path: Callable) -> None:
        """``write_poly_zone`` raises ``NotImplementedError`` naming SZPLT."""
        x, y, faces, left, right = create_poly_octagon()
        with (
            pytest.raises(NotImplementedError, match="SZPLT"),
            tecio.open(str(output_path("err.szplt")), "w") as w,
        ):
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )

    def test_write_fe_zone_points_to_write_poly_zone(
        self, output_path: Callable
    ) -> None:
        """``write_fe_zone`` with a poly type points to ``write_poly_zone``."""
        x, y, nodes = create_FE_tri()
        with (
            pytest.raises(ValueError, match="write_poly_zone"),
            tecio.open(str(output_path("err.szplt")), "w") as w,
        ):
            w.write_fe_zone(zone_type=ZoneType.FEPOLYGON, data=[x, y], node_map=nodes)


class TestWritePolyDat:
    """The ASCII layout that ``write_poly_zone`` produces."""

    def test_writer_uses_tecplot_section_markers(self, output_path: Callable) -> None:
        """Face map sections are labeled with the same ``#`` lines Tecplot writes."""
        x, y, faces, left, right = create_poly_octagon()
        fx, fy, fz, fnodes, fcounts, fleft, fright = create_polyhedron_bconn_fine()
        cx, cy, cz, cnodes, ccounts, cleft, cright, bcounts, belems, bzones = (
            create_polyhedron_bconn_coarse()
        )
        polygon = output_path("write_poly_markers_polygon.dat")
        polyhedron = output_path("write_poly_markers_polyhedron.dat")

        with tecio.open(str(polygon), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )
        with tecio.open(str(polyhedron), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[fx, fy, fz],
                face_nodes=fnodes,
                face_node_counts=fcounts,
                face_left_elements=fleft,
                face_right_elements=fright,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[cx, cy, cz],
                face_nodes=cnodes,
                face_node_counts=ccounts,
                face_left_elements=cleft,
                face_right_elements=cright,
                boundary_connection_counts=bcounts,
                boundary_connection_elements=belems,
                boundary_connection_zones=bzones,
            )

        polygon_text = polygon.read_text()
        text = polyhedron.read_text()
        for marker in ("# face nodes", "# left elements", "# right elements"):
            assert marker in polygon_text
            assert marker in text
        assert "# node count per face" not in polygon_text
        assert "# node count per face" in text
        for marker in (
            "# boundary connection counts",
            "# boundary connection elements",
            "# boundary connection zones",
        ):
            assert marker in text
            assert marker not in polygon_text

    def test_header_keywords(self, output_path: Callable) -> None:
        """The zone header carries the counts the reader needs."""
        x, y, faces, left, right = create_poly_octagon()
        cx, cy, cz, nodes, counts, cleft, cright = create_polyhedron_cube()
        polygon = output_path("write_poly_header_polygon.dat")
        polyhedron = output_path("write_poly_header_polyhedron.dat")

        with tecio.open(str(polygon), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
                face_right_elements=right,
            )
        with tecio.open(str(polyhedron), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYHEDRON,
                data=[cx, cy, cz],
                face_nodes=nodes,
                face_node_counts=counts,
                face_left_elements=cleft,
                face_right_elements=cright,
            )

        text = polygon.read_text()
        assert "Nodes=8, Faces=8, Elements=1, ZONETYPE=FEPolygon" in text
        assert "NumConnectedBoundaryFaces=0, TotalNumBoundaryConnections=0" in text
        assert "TotalNumFaceNodes" not in text  # polygon faces always have 2 nodes
        text = polyhedron.read_text()
        assert "ZONETYPE=FEPolyhedron" in text
        assert "TotalNumFaceNodes=24, NumConnectedBoundaryFaces=0" in text

    def test_ascii_indices_are_one_based(self, output_path: Callable) -> None:
        """ASCII adds one to every index: 0 is no element and -k is boundary face k."""
        fx, fy, ffaces, fleft, fright = create_poly_bconn_fine()
        cx, cy, cfaces, cleft, cright, counts, elems, zones = create_poly_bconn_coarse()
        path = output_path("write_poly_ascii.dat")
        with tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[fx, fy],
                face_nodes=ffaces,
                face_left_elements=fleft,
                face_right_elements=fright,
            )
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[cx, cy],
                face_nodes=cfaces,
                face_left_elements=cleft,
                face_right_elements=cright,
                boundary_connection_counts=counts,
                boundary_connection_elements=elems,
                boundary_connection_zones=zones,
            )

        lines = path.read_text().splitlines()
        # Last zone: the boundary references -2 and -3 are written as -1 and -2.
        start = lines.index("# right elements", len(lines) // 2)
        right = [int(v) for v in lines[start + 1].split()]
        assert min(right) == -2
        elems_line = lines.index("# boundary connection elements")
        assert min(int(v) for v in lines[elems_line + 1].split()) == 1

    def test_failed_first_zone_leaves_no_file(self, output_path: Callable) -> None:
        """A rejected first zone does not leave a partial file behind."""
        x, y, faces, left, right = create_poly_octagon()
        path = output_path("write_poly_failed.dat")
        with contextlib.suppress(ValueError), tecio.open(str(path), "w") as w:
            w.write_poly_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                face_nodes=faces,
                face_left_elements=left,
            )
        assert not path.exists()


class TestWritePolyPlt:
    """PLT output that can be compared byte for byte with the TecIO classic API."""

    def test_write_poly_matches_tecio_classic_output(
        self, output_path: Callable
    ) -> None:
        """``write_poly_zone`` issues the same calls as a TecIO classic program.

        A reference file is written with ``libtecio`` directly, read back, and copied
        through ``write_poly_zone``. The two files must be identical.
        """
        from tecio._io import _copy_zones

        zones = [create_poly_bconn_fine(), create_poly_bconn_coarse()]
        reference = output_path("write_poly_reference.plt")
        libtecio.tecini142(
            str(reference),
            variables=["x", "y"],
            title="poly",
            scratch_dir=str(reference.parent),
        )
        for i, (x, y, faces, left, right, *bconn) in enumerate(zones):
            counts, elems, bzones = bconn if bconn else ([], [], [])
            libtecio.tecpolyzne142(
                zone_title=f"zone {i}",
                zone_type=ZoneType.FEPOLYGON,
                num_nodes=len(x),
                num_faces=len(left),
                num_elements=int(max(left.max(), right.max())) + 1,
                var_sharing=[0, 0],
                value_locations=[ValueLocation.NODAL, ValueLocation.NODAL],
                pas_vars=[0, 0],
                con_sharing=0,
                strand=0,
                solution_time=0.0,
                total_num_face_nodes=faces.size,
                num_connected_boundary_faces=len(counts),
                total_num_boundary_connections=len(elems),
            )
            libtecio.tecdat142(x.astype(np.float64), is_double=True)
            libtecio.tecdat142(y.astype(np.float64), is_double=True)
            libtecio.tecpolyface142(
                np.full(len(left), 2, dtype=np.int32),
                (faces.ravel() + 1).astype(np.int32),
                (left + 1).astype(np.int32),
                (right + 1).astype(np.int32),
            )
            if len(counts):
                libtecio.tecpolybconn142(
                    np.asarray(counts, dtype=np.int32),
                    (np.asarray(elems) + 1).astype(np.int32),
                    (np.asarray(bzones) + 1).astype(np.int32),
                )
        libtecio.tecend142()

        copy = output_path("write_poly_copy.plt")
        with (
            tecio.open(str(reference), "r") as r,
            tecio.open(
                str(copy), "w", title=r.title, variables=r.variables, precision="double"
            ) as w,
        ):
            _copy_zones(r, w)
        assert copy.read_bytes() == reference.read_bytes()


# ======================================================================================
# DATAPACKING=POINT: real only in DAT; rejected by SZL/PLT
# ======================================================================================


class TestDatapackingPoint:
    """``DATAPACKING=POINT`` is a real, ASCII-only feature -- verified once as a
    capability split (binary formats reject it) and once as a DAT-only round-trip (kept
    separate from the shared suite above since SZL/PLT have nothing to parametrize
    here).
    """

    @pytest.fixture(params=["szl", "plt"])
    def binary_fmt(self, request) -> str:
        return request.param

    def test_point_rejected_by_binary_formats(
        self, binary_fmt: str, output_path: Callable
    ) -> None:
        """SZL and PLT raise NotImplementedError for DATAPACKING=POINT."""
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        path = _path(output_path, binary_fmt, "point_rejected")

        with pytest.raises(NotImplementedError):
            with tecio.open(str(path), "w") as w:
                w.write_ordered_zone(
                    data=[x], variables=["x"], datapacking=DataPacking.POINT
                )

    def test_dat_point_matches_block(self, output_path: Callable) -> None:
        """DAT: POINT and BLOCK packing produce identical read-back values."""
        i, j = 6, 7
        x, y, _ = create_ordered((i, j, 1))
        c = scalar_field(x.squeeze(-1), y.squeeze(-1)).astype(np.float64)
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        path_block = output_path("dat_ijk_packing_block.dat")
        path_point = output_path("dat_ijk_packing_point.dat")

        for path, packing in ((path_block, "BLOCK"), (path_point, "POINT")):
            with tecio.open(str(path), "w") as w:
                w.write_ordered_zone(
                    data=[x, y, c], variables=["x", "y", "c"], datapacking=packing
                )

        with (
            tecio.open(str(path_block), "r") as rb,
            tecio.open(str(path_point), "r") as rp,
        ):
            for vi in range(3):
                np.testing.assert_allclose(
                    rb.zones[0].variables[vi].values.ravel(),
                    rp.zones[0].variables[vi].values.ravel(),
                    rtol=_RTOL_DAT,
                )

    def test_dat_point_invalid_datapacking_raises(self, output_path: Callable) -> None:
        """Unrecognised datapacking string raises ValueError immediately."""
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        path = output_path("dat_bad_packing.dat")

        with pytest.raises(ValueError, match="datapacking"):
            with tecio.open(str(path), "w") as w:
                w.write_ordered_zone(data=[x], variables=["x"], datapacking="CSV")


# ======================================================================================
# Precision: SZL/DAT share one override rule, PLT has whole file spec
# ======================================================================================


class TestPrecisionOverride:
    """``precision`` semantics.

    SZL and DAT share one rule (explicit precision overrides only
    floating-point-inferred variables; integers are always preserved) but differ in
    their *default* (SZL: no override at all; DAT: FLOAT/single). PLT has a
    fundamentally different rule (applies to every variable, including integers, with no
    automatic mode) forced by a real, confirmed constraint of the classic API -- not a
    design choice, so it's tested separately rather than parametrized alongside SZL/DAT.
    """

    @pytest.fixture(params=["szl", "dat"])
    def floating_only_fmt(self, request) -> str:
        return request.param

    def test_precision_overrides_floats_only(
        self, floating_only_fmt: str, output_path: Callable
    ) -> None:
        """Explicit precision overrides floating variables but never integers.

        Demonstrates:
        - A ``float64`` variable is downcast under ``precision=FLOAT``
        - A ``float32`` variable is upcast under ``precision=DOUBLE``
        - An ``int32`` "CPU number"-style variable stays INT32 regardless of which
          *precision* is requested
        """
        fmt = floating_only_fmt
        n = 6
        x_f32 = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c_f64 = np.sin(x_f32.astype(np.float64))
        cpu_id = np.array([100, 200, 300, 400, 500, 600], dtype=np.int32)

        for precision in (DataType.FLOAT, DataType.DOUBLE):
            path = _path(output_path, fmt, f"precision_override_{precision.name}")
            with tecio.open(str(path), "w", precision=precision) as w:
                w.write_ordered_zone(
                    data=[x_f32, c_f64, cpu_id],
                    variables=["x_f32", "c_f64", "cpu_id"],
                )

            with tecio.open(str(path), "r") as r:
                zone = r.zones[0]
                assert zone.variables[0].data_type == _expected_dtype(
                    fmt, x_f32, precision
                )
                assert zone.variables[1].data_type == _expected_dtype(
                    fmt, c_f64, precision
                )
                assert zone.variables[2].data_type == DataType.INT32, (
                    f"cpu_id must stay INT32 under precision={precision.name}"
                )
                np.testing.assert_allclose(
                    zone.variables[1].values.ravel(),
                    c_f64,
                    rtol=_rtol_for_precision(precision),
                )
                np.testing.assert_array_equal(zone.variables[2].values.ravel(), cpu_id)

    def test_invalid_precision_string_raises(
        self, floating_only_fmt: str, output_path: Callable
    ) -> None:
        """An unrecognized precision string raises ValueError immediately."""
        fmt = floating_only_fmt
        path = _path(output_path, fmt, "precision_invalid")

        with pytest.raises(ValueError, match="precision"):
            tecio.open(str(path), "w", precision="triple")

    def test_precision_string_aliases(
        self, floating_only_fmt: str, output_path: Callable
    ) -> None:
        """'single'/'double' strings work the same as the DataType enum."""
        fmt = floating_only_fmt
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        for alias, expected in (
            ("single", DataType.FLOAT),
            ("double", DataType.DOUBLE),
        ):
            path = _path(output_path, fmt, f"precision_alias_{alias}")
            with tecio.open(str(path), "w", precision=alias) as w:
                w.write_ordered_zone(data=[x], variables=["x"])
            with tecio.open(str(path), "r") as r:
                assert r.zones[0].variables[0].data_type == expected

    def test_dat_default_precision_downcasts_floats(
        self, output_path: Callable
    ) -> None:
        """DAT's default precision (FLOAT/single) downcasts float64 input.

        This is the key behavior that changed once DAT gained real DT= support: DAT's
        default resolves to FLOAT.
        """
        x = np.array([1.0 / 3.0, 2.0 / 3.0, np.pi], dtype=np.float64)
        path = output_path("dat_default_precision.dat")

        with tecio.open(str(path), "w") as w:  # no precision= -- uses the default
            w.write_ordered_zone(data=[x], variables=["x"])

        with tecio.open(str(path), "r") as r:
            var = r.zones[0].variables[0]
            assert var.data_type == DataType.FLOAT
            # Downcast to float32 precision, not full float64 fidelity.
            np.testing.assert_allclose(var.values.ravel(), x, rtol=_RTOL_F32)

    def test_dat_precision_double_full_fidelity(self, output_path: Callable) -> None:
        """DAT: precision=DOUBLE gives full float64 round-trip fidelity."""
        x = np.array([1.23456789012345678, 2.34567890123456789], dtype=np.float64)
        path = output_path("dat_precision_double.dat")

        with tecio.open(str(path), "w", precision=DataType.DOUBLE) as w:
            w.write_ordered_zone(data=[x], variables=["x"])

        with tecio.open(str(path), "r") as r:
            assert r.zones[0].variables[0].data_type == DataType.DOUBLE
            np.testing.assert_allclose(
                r.zones[0].variables[0].values.ravel(), x, rtol=1e-15
            )

    def test_szl_precision_none_is_fully_automatic(self, output_path: Callable) -> None:
        """SZL: precision=None (the default) preserves every dtype exactly."""
        x_f32 = np.array([1.0, 2.0], dtype=np.float32)
        c_f64 = np.array([3.0, 4.0], dtype=np.float64)
        cpu_id = np.array([7, 8], dtype=np.int32)
        path = output_path("szl_precision_none.szplt")

        with tecio.open(str(path), "w") as w:
            assert w.precision is None
            w.write_ordered_zone(
                data=[x_f32, c_f64, cpu_id], variables=["x", "c", "cpu"]
            )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.variables[0].data_type == DataType.FLOAT
            assert zone.variables[1].data_type == DataType.DOUBLE
            assert zone.variables[2].data_type == DataType.INT32


# ======================================================================================
# SZL-only: incremental flush (tec_file_writer_flush)
# ======================================================================================


class TestSZLFlush:
    """``flush=True`` on the SZL zone writers to incrementally release memory.

    Checks for the six temporary intermediate files TecIO writes to disk the moment
    ``tecFileWriterFlush`` is called -- ``<path>.szhdr``, ``.szdat``, ``.szaux``,
    ``.sztxt``, ``.szgeo``, ``.szlab`` (per the Data Format Guide's ``szcombine``
    section) while the writer is still open, then confirms the final joined file matches
    one not using the ``flush=True`` flag.
    """

    # Suffixes szcombine expects to find and join
    _INTERMEDIATE_SUFFIXES = (
        ".szhdr",
        ".szdat",
        ".szaux",
        ".sztxt",
        ".szgeo",
        ".szlab",
    )

    def test_flush_creates_and_joins_intermediate_files(
        self, output_path: Callable
    ) -> None:
        """``flush=True`` produces ``.sz*`` intermediate files, then close()
        joins and removes them.

        Demonstrates:
        - Calling ``write_ordered_zone(..., flush=True)`` leaves the six ``<path>.sz*``
          intermediate files on disk *while the writer is still open* -- the same files
          an external solver's ``TECFLUSH142``/``tecFileWriterFlush`` calls would leave
          for ``szcombine`` to join later.
        - ``close()`` performs the equivalent of ``szcombine <path> --cleanup``: the
          final ``.szplt`` appears and the intermediate files are gone.
        - The joined file still round-trips to the correct zone data.
        """
        path = output_path("szl_flush_intermediate_files.szplt")
        temp_files = [Path(f"{path}{suffix}") for suffix in self._INTERMEDIATE_SUFFIXES]

        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        z = z.astype(np.float64)
        c = scalar_field(x, y, z).astype(np.float64)

        w = tecio.open(str(path), "w", variables=["x", "y", "z", "c"])
        try:
            w.write_ordered_zone(data=[x, y, z, c], flush=True)

            # With the writer is still open check these files exist on disk
            for f in temp_files:
                assert f.exists(), (
                    f"expected intermediate file {f} after flush=True; "
                    "flush may not have actually run"
                )
        finally:
            w.close()

        # Check that intermediate files are removed on join
        assert path.exists()
        for f in temp_files:
            assert not f.exists(), f"intermediate file {f} should be gone after close()"

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 1
            zone = r.zones[0]
            assert zone.dimensions == (i, j, k)
            np.testing.assert_allclose(
                zone.variables[3].values.ravel(), c.ravel(), rtol=_RTOL_F64
            )


# ======================================================================================
# Entry point
# ======================================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
