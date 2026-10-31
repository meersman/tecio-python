#!/usr/bin/env python3
"""pytest tests for :class:`tecio.{szl,plt,dat}.Read` — shared reader conformance suite.

All ``Read`` classes are built to an identical core API, so most of this suite runs
the same test body against all three formats, parametrized by ``fmt``.

Note
----
DAT's original test file hand-authored small ASCII fixtures inline (``textwrap.dedent``)
for its container/get_array/passive tests instead of relying on the external
``Onera.dat`` file, and had an entire extra ``TestPointFormat`` class with no SZL/PLT
equivalent. That's a *better* pattern than depending on an external fixture where
practical: no fixture file to keep in sync, and exact values to assert against instead
of the generic/loose checks the ``Onera``-based tests use. This suite:

* Ports that pattern to :func:`test_get_array_passive` for every format (previously
  SZL/PLT round-tripped through their own ``Write`` class here, which is exactly the
  "not independent of the writer" gap called out above).
* Keeps DAT's ``TestPointFormat`` as DAT-only (POINT is a real, DAT-only capability —
  same precedent as ``test_write.py``'s ``TestDatapackingPoint``).
* Leaves ``TestReadDump``/``TestContainers``/``test_get_array_matches_variable_values``
  on the external ``Onera.*`` fixture, parametrized across formats. This assumes
  ``Onera.szplt``, ``Onera.plt``, and ``Onera.dat`` all exist side by side -- **verify
  this before trusting these tests**; the ``onera_file`` fixture below skips (rather
  than failing) a format whose fixture is missing, which will silently under-test until
  the file is added. Hand- authoring equivalent binary SZL/PLT fixtures (packing bytes
  directly, the way you'd construct a minimal ``.plt``/``.szplt`` by hand) would remove
  that external dependency the same way DAT's ASCII fixtures do, but is meaningfully
  more work per format and is left as a follow-up rather than attempted in this initial
  pass.
* Poly (FEPOLYHEDRON) zones are read from ``Sphere.plt`` and ``Sphere.dat``. Polygon
  zones and boundary connnection attributes not explicit in ``Sphere.plt`` are covered
  by writing and reading back in ``test_write.py``.

Run directly:

    $ python tests/test_read.py -v
"""

# ruff: noqa: E501, SIM117

import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import tecio
from tecio import DataPacking, DataType, ValueLocation, ZoneType

# ======================================================================================
# Cross-format helpers
# ======================================================================================

FORMATS: list[str] = ["szl", "plt", "dat"]
_EXTENSIONS: dict[str, str] = {"szl": "szplt", "plt": "plt", "dat": "dat"}


@pytest.fixture(params=FORMATS)
def fmt(request) -> str:
    """Parametrize a test across every reader format (szl, plt, dat)."""
    return request.param


@pytest.fixture
def onera_file(fmt: str, tests_dir: Path) -> Path:
    """Resolve the per-format Onera reference fixture.

    Skips (rather than fails) a format whose fixture file doesn't exist, so
    this suite degrades gracefully until ``Onera.plt``/``Onera.dat`` are
    confirmed present alongside ``Onera.szplt`` -- see module docstring.
    """
    path = tests_dir / f"Onera.{_EXTENSIONS[fmt]}"
    if not path.exists():
        pytest.skip(f"No Onera fixture for format {fmt!r} at {path}")
    return path


def _path(output_path: Callable, fmt: str, name: str) -> Path:
    return output_path(f"{name}.{_EXTENSIONS[fmt]}")


def _write_text_fixture(output_path: Callable, name: str, text: str) -> Path:
    """Write a hand-authored ASCII (.dat) fixture and return its path."""
    path = output_path(name)
    path.write_text(text, encoding="utf-8")
    return path


# ======================================================================================
# Full-file dump (Onera reference fixture)
# ======================================================================================


class TestReadDump:
    """Dump every header and data field for the shared Onera fixture."""

    def test_read_dump(self, fmt: str, onera_file: Path) -> None:
        """Print out all headers and data fields for the Onera fixture.

        Demonstrates:
        - Top-level metadata: file_type, title, num_vars, variables, num_zones
        - Dataset- and variable-level auxiliary data access
        - Per-zone metadata: title, zone_type, datapacking, solution_time,
          strand_id, and zone-level aux data
        - Per-variable metadata: name, data_type, value_location,
          is_enabled/is_passive, shared_zone, num_values, and the raw array
        - Connectivity (node_map) for FE zones
        """
        np.set_printoptions(threshold=100)

        r = tecio.open(str(onera_file), "r")

        print("\nFile Record")
        print("=" * 70)
        print(f"File Type         : {r.file_type}")
        print(f"Dataset Title     : {r.title}")
        print(f"Num Vars          : {r.num_vars}")
        print(f"Variables         : {r.variables}")
        print(f"Num Zones         : {r.num_zones}")
        print(f"Dataset Aux Items : {r.num_auxdata_items}")

        print("\n\nDataset Auxiliary Data")
        print("-" * 70)
        if len(r.auxdata) > 0:
            print(f"Dataset Aux Data  : {dict(r.auxdata)}")
            for name, value in r.auxdata.items():
                print(f"  {name:>15} : {value}")

        print("\n\nVariable Auxiliary Data")
        print("-" * 70)
        for i in range(r.num_vars):
            var_aux = r.get_var_auxdata(i)
            if len(var_aux) > 0:
                print(f"Var {i:3} Aux Data  : {dict(var_aux)}")
                for name, value in var_aux.items():
                    print(f"  {name:>15} : {value}")

        print("\n\nZone Record")
        print("-" * 70)
        for i in range(r.num_zones):
            zone = r.zones[i]
            print(f"\nZone {i:3}")
            print(f"  Title           : {zone.title}")
            print(f"  Zone Type       : {zone.zone_type}")
            print(f"  Datapacking     : {zone.datapacking}")
            print(f"  Is Enabled      : {zone.is_enabled()}")
            print(f"  Solution Time   : {zone.solution_time}")
            print(f"  Strand ID       : {zone.strand_id}")

            if len(zone.auxdata) > 0:
                print(f"  Zone Aux Data   : {dict(zone.auxdata)}")
                for name, value in zone.auxdata.items():
                    print(f"  {name:>15} : {value}")

            for j in range(r.num_vars):
                var = zone.variables[j]
                print(f"  Variable {j:3}")
                print(f"    Name          : {var.name}")
                print(f"    Data Type     : {var.data_type}")
                print(f"    Is Enabled    : {var.is_enabled()}")
                print(f"    Location      : {var.value_location}")
                print(f"    Is Passive    : {var.is_passive()}")
                print(f"    Shared Zone   : {var.shared_zone}")
                print(f"    Num Values    : {var.num_values}")

                value_str = np.array2string(
                    var.values, prefix="    Values        : ", separator=", "
                )
                print(f"    Values        : {value_str}")

            if zone.zone_type != ZoneType.ORDERED:
                print(f"  Node Map Shape  : {zone.node_map.shape}")
                value_str = np.array2string(
                    zone.node_map, prefix="  Connectivity    : ", separator=", "
                )
                print(f"  Connectivity    : {value_str}")


# ======================================================================================
# ZoneList / VariableList containers (Onera reference fixture)
# ======================================================================================


class TestContainers:
    """``ZoneList`` / ``VariableList`` container behaviour."""

    def test_zone_and_variable_containers(self, fmt: str, onera_file: Path) -> None:
        """Verify ``zone``/``variable`` return the new container types.

        Demonstrates:
        - ``Read.zones`` is a :class:`tecio.ZoneList`: an int index returns a
          ``ReadZone``; a slice returns another ``ZoneList`` of the same kind
        - ``ReadZone.variables`` is a :class:`tecio.VariableList`: index by
          0-based position or by exact, case-sensitive name
        - Unknown name -> ``KeyError``; out-of-range index -> ``IndexError``
        """
        r = tecio.open(str(onera_file), "r")

        assert isinstance(r.zones, tecio.ZoneList)
        assert len(r.zones) == r.num_zones

        if r.num_zones >= 2:
            sub = r.zones[0:2]
            assert isinstance(sub, tecio.ZoneList)
            assert len(sub) == 2
            assert sub[0].title == r.zones[0].title
            assert sub[1].title == r.zones[1].title

        zone = r.zones[0]

        assert isinstance(zone.variables, tecio.VariableList)
        assert len(zone.variables) == r.num_vars
        assert zone.variables.names() == r.variables

        first_name = r.variables[0]
        assert zone.variables[0].name == zone.variables[first_name].name == first_name

        # Name lookup is exact and case-sensitive.
        swapped = first_name.swapcase()
        if swapped != first_name:
            with pytest.raises(KeyError):
                zone.variables[swapped]

        with pytest.raises(KeyError):
            zone.variables["__not_a_real_variable__"]

        with pytest.raises(IndexError):
            zone.variables[r.num_vars + 10]

    def test_zone_and_variable_containers_hand_authored(
        self, output_path: Callable
    ) -> None:
        """DAT-only: exact-value container check against a hand-authored fixture.

        Demonstrates:
        - A hand-authored two-zone BLOCK file (rather than the opaque Onera
          fixture) so zone slicing can be checked against known titles and
          exact zone count -- something the generic Onera-based check above
          can't do without knowing the fixture's contents in advance
        """
        two_zone_dat = textwrap.dedent("""\
            TITLE     = "two_zone_test"
            VARIABLES = "x" "y" "p"
            ZONE T="Zone1"
             STRANDID=0, SOLUTIONTIME=0.0
             I=4, J=1, K=1, ZONETYPE=Ordered
             DATAPACKING=BLOCK
            0.0 1.0 2.0 3.0
            0.0 1.0 2.0 3.0
            10.0 20.0 30.0 40.0
            ZONE T="Zone2"
             STRANDID=0, SOLUTIONTIME=1.0
             I=4, J=1, K=1, ZONETYPE=Ordered
             DATAPACKING=BLOCK
            4.0 5.0 6.0 7.0
            4.0 5.0 6.0 7.0
            50.0 60.0 70.0 80.0
            """)
        path = _write_text_fixture(
            output_path, "read_containers_hand.dat", two_zone_dat
        )

        r = tecio.open(str(path), "r")

        assert r.num_vars == 3
        assert r.num_zones == 2

        sub = r.zones[0:2]
        assert sub[0].title == "Zone1"
        assert sub[1].title == "Zone2"

        zone = r.zones[0]
        assert zone.variables.names() == ["x", "y", "p"]
        assert zone.variables[0].name == zone.variables["x"].name == "x"

        with pytest.raises(KeyError):
            zone.variables["X"]  # case-sensitive
        with pytest.raises(IndexError):
            zone.variables[10]


# ======================================================================================
# ReadZone.get_array
# ======================================================================================


class TestGetArray:
    """``ReadZone.get_array`` scalar, list, and passive-variable behaviour."""

    def test_get_array_matches_variable_values(
        self, fmt: str, onera_file: Path
    ) -> None:
        """Verify ``get_array`` parity with ``variable[...].values``.

        Demonstrates:
        - Scalar index / scalar name both return the same array as the
          explicit ``zone.variables[key].values`` path
        - A list of names returns a tuple, in order -- never a bare array,
          even for a single-element list
        - Bad keys raise the same errors as ``VariableList``
        """
        r = tecio.open(str(onera_file), "r")
        zone = r.zones[0]
        names = r.variables

        for i, name in enumerate(names[: min(3, len(names))]):
            expected = zone.variables[i].values
            if expected is None:
                continue
            np.testing.assert_array_equal(zone.get_array(i), expected)
            np.testing.assert_array_equal(zone.get_array(name), expected)

        if len(names) >= 2:
            pair = zone.get_array(names[:2])
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            for arr, name in zip(pair, names[:2], strict=True):
                expected = zone.variables[name].values
                if expected is None:
                    assert arr is None
                else:
                    np.testing.assert_array_equal(arr, expected)

        one = zone.get_array([names[0]])
        assert isinstance(one, tuple)
        assert len(one) == 1

        with pytest.raises(KeyError):
            zone.get_array("__not_a_real_variable__")

        with pytest.raises(IndexError):
            zone.get_array(r.num_vars + 10)

    def test_get_array_passive(self, fmt: str, output_path: Callable) -> None:
        """``get_array`` returns ``None`` for a passive variable, by index or name.

        Hand-authored per format (not round-tripped through ``Write``) so
        this is a reader-only test -- it can't pass merely because the
        writer and reader agree with each other. DAT's fixture is plain
        text; SZL/PLT still go through ``Write`` here since hand-authoring
        their binary layout is out of scope for this pass (see module
        docstring), which is the one place this test isn't yet fully
        writer-independent for those two formats.
        """
        if fmt == "dat":
            passive_dat = textwrap.dedent("""\
                TITLE     = "dat_passive_test"
                VARIABLES = "x" "y" "p"
                ZONE T="ZonePassive"
                 STRANDID=0, SOLUTIONTIME=0.0
                 I=4, J=1, K=1, ZONETYPE=Ordered
                 DATAPACKING=BLOCK
                 PASSIVEVARLIST=[3]
                0.0 1.0 2.0 3.0
                0.0 1.0 2.0 3.0
                """)
            path = _write_text_fixture(
                output_path, "read_get_array_passive.dat", passive_dat
            )
            x = np.array([0.0, 1.0, 2.0, 3.0])
        else:
            x = np.array([0.0, 1.0, 2.0, 3.0])
            y = np.array([0.0, 1.0, 4.0, 9.0])
            path = _path(output_path, fmt, "read_get_array_passive")
            with tecio.open(str(path), "w") as w:
                w.write_ordered_zone(
                    data=[x, y],
                    variables=["x", "y", "p"],
                    passive_vars={"p"},
                )

        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]

            assert zone.variables["p"].is_passive()
            assert zone.get_array("p") is None
            assert zone.get_array(2) is None

            np.testing.assert_allclose(zone.get_array("x").ravel(), x)

            mixed = zone.get_array(["x", "p"])
            assert isinstance(mixed, tuple)
            np.testing.assert_allclose(mixed[0].ravel(), x)
            assert mixed[1] is None


# ======================================================================================
# DATAPACKING=POINT parsing (DAT-only)
# ======================================================================================


class TestDatPointFormat:
    """Hand-authored ``DATAPACKING=POINT`` fixtures, mimicking third-party writers.

    Exercises the reader against POINT-format text this project's own
    writer may never happen to produce (different whitespace, ordering,
    etc. than a real external tool might use) -- exactly the kind of
    coverage a write-then-read round trip structurally can't provide.
    """

    def test_read_point_ordered(self, output_path: Callable) -> None:
        """Read a hand-authored DATAPACKING=POINT ordered zone."""
        point_dat = textwrap.dedent("""\
            TITLE     = "point_test"
            VARIABLES = "x" "y" "p"
            ZONE T="Zone1"
             STRANDID=0, SOLUTIONTIME=0.0
             I=4, J=1, K=1, ZONETYPE=Ordered
             DATAPACKING=POINT
            0.0\t1.0\t10.0
            1.0\t2.0\t20.0
            2.0\t3.0\t30.0
            3.0\t4.0\t40.0
            """)
        expected_x = np.array([0.0, 1.0, 2.0, 3.0])
        expected_y = np.array([1.0, 2.0, 3.0, 4.0])
        expected_p = np.array([10.0, 20.0, 30.0, 40.0])

        path = _write_text_fixture(output_path, "read_point_ordered.dat", point_dat)
        r = tecio.open(str(path), "r")

        assert r.num_vars == 3
        assert r.num_zones == 1

        zone = r.zones[0]
        assert zone.zone_type == ZoneType.ORDERED
        assert zone.datapacking == DataPacking.POINT
        assert zone.dimensions == (4, 1, 1)

        np.testing.assert_allclose(zone.variables[0].values.ravel(), expected_x)
        np.testing.assert_allclose(zone.variables[1].values.ravel(), expected_y)
        np.testing.assert_allclose(zone.variables[2].values.ravel(), expected_p)

    def test_read_point_fe(self, output_path: Callable) -> None:
        """Read a hand-authored DATAPACKING=POINT FE triangle zone."""
        point_dat = textwrap.dedent("""\
            TITLE     = "point_fe_test"
            VARIABLES = "x" "y" "c"
            ZONE T="FE_Tri_Point"
             STRANDID=0, SOLUTIONTIME=0.0
             Nodes=4, Elements=2, ZONETYPE=FETriangle
             DATAPACKING=POINT
            0.0\t0.0\t0.1
            1.0\t0.0\t0.2
            1.0\t1.0\t0.3
            0.0\t1.0\t0.4
            1 2 3
            1 3 4
            """)
        expected_x = np.array([0.0, 1.0, 1.0, 0.0])
        expected_c = np.array([0.1, 0.2, 0.3, 0.4])
        expected_conn = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        path = _write_text_fixture(output_path, "read_point_fe.dat", point_dat)
        r = tecio.open(str(path), "r")

        assert r.num_vars == 3
        assert r.num_zones == 1

        zone = r.zones[0]
        assert zone.zone_type == ZoneType.FETRIANGLE
        assert zone.datapacking == DataPacking.POINT
        assert zone.num_nodes == 4
        assert zone.num_elements == 2

        np.testing.assert_allclose(zone.variables[0].values, expected_x)
        np.testing.assert_allclose(zone.variables[2].values, expected_c)
        np.testing.assert_array_equal(zone.node_map, expected_conn)

    def test_read_point_mixed_cc(self, output_path: Callable) -> None:
        """Read a POINT zone with mixed nodal and cell-centred variables."""
        point_dat = textwrap.dedent("""\
            TITLE     = "point_cc_test"
            VARIABLES = "x" "p" "rho"
            ZONE T="CC_Point"
             STRANDID=0, SOLUTIONTIME=0.0
             I=4, J=1, K=1, ZONETYPE=Ordered
             DATAPACKING=POINT
             VARLOCATION=([3]=CELLCENTERED)
            0.0\t1.0
            1.0\t2.0
            2.0\t3.0
            3.0\t4.0
            10.0
            20.0
            30.0
            """)
        path = _write_text_fixture(output_path, "read_point_mixed_cc.dat", point_dat)
        r = tecio.open(str(path), "r")

        assert r.num_vars == 3
        zone = r.zones[0]
        assert zone.datapacking == DataPacking.POINT

        x_var = zone.variables[0]
        p_var = zone.variables[1]
        rho_var = zone.variables[2]

        assert x_var.value_location == ValueLocation.NODAL
        assert p_var.value_location == ValueLocation.NODAL
        assert rho_var.value_location == ValueLocation.CELL_CENTERED

        np.testing.assert_allclose(x_var.values.ravel(), [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(p_var.values.ravel(), [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(rho_var.values.ravel(), [10.0, 20.0, 30.0])

    def test_read_point_ordered_no_dimensions(self, output_path: Callable) -> None:
        """Legacy ``F=POINT`` with no I/J/K infers the point count from data.

        Some legacy exporters write a bare zone header relying on the reader
        to count data rows itself, rather than declaring I/J/K explicitly.
        Two zones in one file confirm the inferred read correctly stops at
        the next ZONE keyword rather than over-consuming.
        """
        point_dat = textwrap.dedent("""\
            TITLE     = "legacy_no_dims"
            VARIABLES = "x" "y"
            ZONE T="First", F=POINT
            1.0 10.0
            2.0 20.0
            3.0 30.0
            ZONE T="Second", F=POINT
            100.0 1000.0
            200.0 2000.0
            """)
        path = _write_text_fixture(
            output_path, "read_point_ordered_no_dims.dat", point_dat
        )
        r = tecio.open(str(path), "r")

        assert r.num_zones == 2

        first = r.zones[0]
        assert first.zone_type == ZoneType.ORDERED
        assert first.datapacking == DataPacking.POINT
        assert first.dimensions == (3, 1, 1)
        np.testing.assert_allclose(first.variables[0].values.ravel(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            first.variables[1].values.ravel(), [10.0, 20.0, 30.0]
        )

        second = r.zones[1]
        assert second.dimensions == (2, 1, 1)
        np.testing.assert_allclose(second.variables[0].values.ravel(), [100.0, 200.0])
        np.testing.assert_allclose(second.variables[1].values.ravel(), [1000.0, 2000.0])

    def test_read_point_ordered_no_dimensions_malformed_raises(
        self, output_path: Callable
    ) -> None:
        """A truncated final row raises a clear error rather than misreading."""
        point_dat = textwrap.dedent("""\
            TITLE     = "legacy_malformed"
            VARIABLES = "x" "y"
            ZONE F=POINT
            1.0 10.0
            2.0 20.0
            3.0
            """)
        path = _write_text_fixture(
            output_path, "read_point_ordered_malformed.dat", point_dat
        )
        with pytest.raises(ValueError, match="not evenly divisible"):
            tecio.open(str(path), "r")


# ======================================================================================
# Poly (FEPOLYHEDRON) zones (PLT and DAT only)
# ======================================================================================

#: Formats that can hold poly zones.
POLY_FORMATS: list[str] = ["plt", "dat"]


@pytest.fixture
def sphere_file(fmt: str, tests_dir: Path) -> Path:
    """Resolve the per-format Sphere reference fixture (polyhedral zone)."""
    path = tests_dir / f"Sphere.{_EXTENSIONS[fmt]}"
    if not path.exists():
        pytest.skip(f"No Sphere fixture for format {fmt!r} at {path}")
    return path


@pytest.mark.parametrize("fmt", POLY_FORMATS)
class TestReadPoly:
    """Reading the polyhedral Sphere fixture."""

    def test_read_poly_metadata(self, fmt: str, sphere_file: Path) -> None:
        """Dataset, zone and variable metadata of a FEPOLYHEDRON zone.

        Demonstrates:
        - ``TecplotPolyZoneReader`` counts: ``num_nodes``, ``num_elements`` and
          ``num_faces``, plus the boundary connection counts
        - Poly zones are always BLOCK packed
        """
        with tecio.open(str(sphere_file), "r") as r:
            assert r.title == "Finite-Element Data"
            assert r.variables == ["X", "Y", "Z", "Cycling"]
            assert r.num_zones == 1

            zone = r.zones[0]
            assert isinstance(zone, tecio.TecplotPolyZoneReader)
            assert zone.title == "Sphere"
            assert zone.zone_type == ZoneType.FEPOLYHEDRON
            assert zone.datapacking == DataPacking.BLOCK
            assert zone.strand_id == 1
            assert zone.solution_time == 0.0
            assert zone.shared_connectivity is None
            assert zone.num_nodes == 5202
            assert zone.num_elements == 1380
            assert zone.num_faces == 7238
            assert zone.num_boundary_faces == 0
            assert zone.num_boundary_connections == 0
            for var in zone.variables:
                assert var.data_type == DataType.DOUBLE
                assert var.value_location == ValueLocation.NODAL

    def test_read_poly_face_map(self, fmt: str, sphere_file: Path) -> None:
        """Face arrays: sizes, zero-based indices and the first and last entries.

        The expected values were read from the text of ``Sphere.dat``, which stores the
        same arrays with one-based indices.

        Demonstrates:
        - ``face_nodes`` is flat and ``face_node_counts`` gives the nodes of each face
        - ``face_node_offsets`` has ``num_faces + 1`` entries and slices ``face_nodes``
        - ``face_left_elements``/``face_right_elements`` are element indices, ``-1`` for no element
        """
        with tecio.open(str(sphere_file), "r") as r:
            zone = r.zones[0]

            counts = zone.face_node_counts
            assert counts.shape == (7238,)
            assert counts.sum() == 35145
            np.testing.assert_array_equal(counts[:10], [4, 4, 4, 4, 4, 4, 6, 6, 4, 4])
            np.testing.assert_array_equal(counts[-5:], 4)
            np.testing.assert_array_equal(
                np.bincount(counts), [0, 0, 0, 4, 3271, 2067, 1583, 289, 23, 1]
            )

            nodes = zone.face_nodes
            assert nodes.shape == (35145,)
            np.testing.assert_array_equal(
                nodes[:8], [398, 397, 390, 391, 2033, 398, 391, 2027]
            )
            np.testing.assert_array_equal(nodes[-5:], [4246, 5011, 4719, 4816, 4814])
            assert nodes.min() == 0
            assert nodes.max() == zone.num_nodes - 1

            offsets = zone.face_node_offsets
            assert offsets.shape == (7239,)
            assert offsets[0] == 0
            assert offsets[-1] == 35145
            np.testing.assert_array_equal(np.diff(offsets), counts)
            np.testing.assert_array_equal(
                nodes[offsets[1] : offsets[2]], [2033, 398, 391, 2027]
            )

            left, right = zone.face_left_elements, zone.face_right_elements
            assert left.shape == right.shape == (7238,)
            np.testing.assert_array_equal(left[:10], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
            np.testing.assert_array_equal(left[-5:], [1366, 1367, 1368, 1374, 1376])
            np.testing.assert_array_equal(
                right[:10], [176, 213, 30, 216, 264, 32, 598, -1, 254, 169]
            )
            np.testing.assert_array_equal(right[-5:], [1369, 1378, 1371, 1376, 1377])

    def test_read_poly_face_map_is_consistent(
        self, fmt: str, sphere_file: Path
    ) -> None:
        """Every element is a closed polyhedron, so the face map must add up."""
        with tecio.open(str(sphere_file), "r") as r:
            zone = r.zones[0]
            left, right = zone.face_left_elements, zone.face_right_elements

            # Left elements are always set, right elements are -1 on the 330 faces
            # that lie on the outer surface.
            assert left.min() == 0
            assert left.max() == 1376
            assert right.min() == -1
            assert right.max() == 1379
            assert np.count_nonzero(right == -1) == 330

            # Every element is referenced, with between 7 and 20 faces each.
            per_element = np.bincount(np.concatenate([left, right[right >= 0]]))
            assert per_element.shape == (1380,)
            assert per_element.min() == 7
            assert per_element.max() == 20
            assert per_element.sum() == 2 * 7238 - 330

    def test_read_poly_variables(self, fmt: str, sphere_file: Path) -> None:
        """Nodal variables of a poly zone have one value per node.

        The expected values were read from the text of ``Sphere.dat``.
        """
        with tecio.open(str(sphere_file), "r") as r:
            zone = r.zones[0]
            x, y, z, cycling = (zone.get_array(name) for name in r.variables)
            for values in (x, y, z, cycling):
                assert values.shape == (5202,)
                assert values.dtype == np.float64

            np.testing.assert_allclose(x[[0, -1]], [0.1545234621, -2.117949963])
            np.testing.assert_allclose(y[[0, -1]], [2.923508644, -0.135193646])
            np.testing.assert_allclose(z[[0, -1]], [-0.5178240538, 2.090382576])
            np.testing.assert_allclose(
                [x.sum(), y.sum(), z.sum()],
                [144.2886312321637, -52.22106074105915, -52.01794189945177],
                rtol=1e-8,
            )
            # "Cycling" counts the nodes from 1.
            np.testing.assert_array_equal(cycling, np.arange(1, 5203))

    def test_read_poly_arrays_are_cached(self, fmt: str, sphere_file: Path) -> None:
        """Counts come from the zone header; face arrays load once, on first use."""
        with tecio.open(str(sphere_file), "r") as r:
            zone = r.zones[0]
            assert zone.num_faces == 7238
            assert "F=7238" in zone._repr_size()
            assert zone._array_cache == {}
            assert zone.face_nodes is zone.face_nodes
            assert zone.face_node_offsets is zone.face_node_offsets
            with pytest.raises(AttributeError):
                zone.num_faces = 3  # type: ignore[misc]


def test_read_poly_plt_matches_dat(tests_dir: Path) -> None:
    """The PLT and DAT exports of the Sphere zone read back the same."""
    with (
        tecio.open(str(tests_dir / "Sphere.plt"), "r") as plt,
        tecio.open(str(tests_dir / "Sphere.dat"), "r") as dat,
    ):
        a, b = plt.zones[0], dat.zones[0]
        for name in (
            "face_nodes",
            "face_node_counts",
            "face_left_elements",
            "face_right_elements",
        ):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        for name in plt.variables:
            np.testing.assert_allclose(a.get_array(name), b.get_array(name), rtol=1e-8)


def test_peek_reports_poly_zone(tests_dir: Path) -> None:
    """``tecio.peek`` lists a poly zone of a PLT file without loading its face map."""
    summary = tecio.peek(str(tests_dir / "Sphere.plt"))
    assert summary.zones[0].zone_type == ZoneType.FEPOLYHEDRON
    assert summary.zones[0].num_elements == 1380


# ======================================================================================
# Entry point
# ======================================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
