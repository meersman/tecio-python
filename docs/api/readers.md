# Readers

{func}`tecio.open` (mode ``'r'``) returns a {class}`~tecio.TecplotSzlReader`,
{class}`~tecio.TecplotPltReader`, or {class}`~tecio.TecplotDatReader` based on
the file extension. All three implement the same interface, this page
documents that interface once; construct a specific class directly only if you
want to bypass the extension-based dispatch.

## What each format keeps in memory

This affects how expensive different operations are, not what you can do with
the reader, the interface below is identical either way.

| Format | On open | On first zone access | Array access |
|---|---|---|---|
| SZL | A live C file handle pointer. | Zone headers resolved via C TecIO function calls. | Live C TecIO function call each time. |
| PLT | Whole header/zone-metadata section parsed eagerly with no file handle kept open. | Output directly from metadata parsed on open. | Read from disk on demand, using offsets recorded at open. |
| DAT | The entire file, header and every array, parsed into memory. | Output directly from in memory data parsed at open. | Already in memory, no further disk access. |

Opening a PLT or DAT file (however large) does one pass over the file; opening
SZL does none, cost is deferred to whatever you actually touch (sub-zone
loaded on demand access is preserved).

## Zones and Variables: `ZoneList` / `VariableList`

`reader.zones` and `zone.variables` return one of two format-agnostic container
types:

```python
with tecio.open("flow.szplt") as r:
    r.zones[0]  # -> a zone reader (Ordered or FE, see below)
    r.zones[1:4]  # -> ZoneList, a sub-range, same kind
    r.zones[0].variables  # -> VariableList
    r.zones[0].variables["x"]  # -> TecplotVariableReader, by exact name
    r.zones[0].variables[2]  # -> TecplotVariableReader, by 0-based index
```

| Class | Returned by | Supports |
|---|---|---|
| {class}`~tecio.ZoneList` | `TecplotReader.zones` | `len()`, iteration, `int` index, `slice` (returns another `ZoneList`) |
| {class}`~tecio.VariableList` | `TecplotZoneReader.variables` | `len()`, iteration, `int` index, exact-name `str` index, `in` |

To output NumPy arrays directly at the zone level, use `get_array`:

```python
p = r.zones[0].get_array("p")  # ndarray | None
p = r.zones[0].get_array(2)  # by 0-based index
x, y, z = r.zones[0].get_array(["x", "y", "z"])  # tuple, for unpacking
```

A single key (index or name) returns one array; a list of names returns a
tuple in the order given. Returns `None` for a passive or shared variable
(shared resolves to the source zone's real data automatically, `None` only
means "no data at all"); raises `KeyError` for an unknown name or
`IndexError` for an out-of-range index. There's deliberately no cross-zone
accessor, to pull one variable across many zones, iterate explicitly so the
outer axis stays in your code:

```python
seq = [z.get_array("p") for z in r.zones]  # list[ndarray | None]
stack = np.stack(seq)  # only once you know the shapes match
```

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: readers

   ZoneList
   VariableList
```

## `TecplotReader`

One open file. Not constructed directly, {func}`tecio.open` returns the
concrete class for you.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: readers

   TecplotReader
```

## Zones: `TecplotZoneReader`, split by topology

Only properties that mean the same thing for *every* zone, regardless of
topology, live on the shared base. Dimensions and connectivity don't: an
ordered zone has no node map, an FE zone has no `I`/`J`/`K`. Each lives only
on its own subclass, and accessing the wrong one raises `AttributeError`
rather than returning `None`:

```python
zone = r.zones[0]
zone.title, zone.zone_type  # always available
if isinstance(zone, tecio.TecplotOrderedZoneReader):
    zone.dimensions  # (I, J, K)
elif isinstance(zone, tecio.TecplotFEZoneReader):
    zone.node_map  # ndarray | None
elif isinstance(zone, tecio.TecplotPolyZoneReader):
    zone.face_nodes  # ndarray
```

`TecplotFEZoneReader` covers the classic finite-element types (and FEMIXED
metadata). FEPOLYGON and FEPOLYHEDRON zones are returned as
`TecplotPolyZoneReader` (PLT v111 and later, and DAT; SZL does not yet support
poly zones).

### Poly zones

Face arrays are returned as stored on disk, 0-based:

| Property | Meaning |
|---|---|
| `face_node_counts` | Nodes per face (all `2` for polygons) |
| `face_node_offsets` | `len(face_nodes)` offsets, length `num_faces + 1` |
| `face_nodes` | Flat 0-based node indices of every face |
| `face_left`, `face_right` | 0-based element per face; `-1` = none, `-(b + 2)` = boundary face `b` |
| `boundary_connection_counts` | Connections per boundary face |
| `boundary_connection_elements` | 0-based elements in other zones or with several neighbors |
| `boundary_connection_zones` | 0-based zone of each connection |

The boundary arrays are empty when the zone has no boundary faces.
`num_nodes`, `num_elements`, `num_faces`, `num_boundary_faces`,
`num_boundary_connections` and `shared_connectivity` are also available. The
left and right elements cover face neighbors, so poly zones have no separate
face-neighbor connections.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: readers

   TecplotZoneReader
   TecplotOrderedZoneReader
   TecplotFEZoneReader
   TecplotPolyZoneReader
```

## `TecplotVariableReader`

One variable's metadata and data within a zone. Not split by value location
(nodal vs. cell-centered) or by the owning zone's topology, neither removes
a property, both only change the shape `get_values()`/`values` returns,
already computed for you from `value_location` and the owning zone.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: readers

   TecplotVariableReader
```

Every concrete variable reader also carries a real, working `var_index`
(0-based, same convention as `zone_index`), not currently promoted to this
shared base class the way `zone_index` was for zones, so it's absent from
the generated page above and doesn't type-check against code written against
this abstract interface, but it's there and correct at runtime, since
`zone.variables[i]` always returns a concrete instance. See {doc}`index`.

## `TecplotAuxDataReader`

A read-only, dict-like view of one auxiliary-data mapping, dataset-, zone-,
or variable-level, same class for all three and every format. Implements
{class}`collections.abc.Mapping`, so `get`, `keys`, `values`, `items`, `in`,
and `len()` all work as expected, plus typed convenience accessors:

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: readers

   TecplotAuxDataReader
```

## Format-specific classes

Constructor and a handful of format quirks; everything else is the shared
interface above.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: readers

   TecplotSzlReader
   TecplotPltReader
   TecplotDatReader
```

:::{versionadded} 0.4.0
Read only `TecplotPltReader` support for several legacy PLT binary versions
(v7.5, v10.7, v11.1). Note that v7.5 PLT files have no fields for passive
variables, strand/ solution-time, cell-centered data, or auxilary data. See
{doc}`legacy_plt_versions` for the full version-by-version comparison.
:::