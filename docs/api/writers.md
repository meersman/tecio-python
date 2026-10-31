# Writers

{func}`tecio.open` returns a {class}`~tecio.TecplotSzlWriter`,
{class}`~tecio.TecplotPltWriter`, or {class}`~tecio.TecplotDatWriter` for
modes ``'w'``/``'x'``, based on the output file extension, or an
{class}`~tecio.AppendWrite`/{class}`~tecio.AppendReadWrite` for modes
``'a'``/``'a+'``. All three concrete writers share the same five-method
lifecycle:

```python
with tecio.open("out.szplt", "w", variables=["x", "y", "p"]) as w:    # 1. open
    w.add_auxdataset_dict({"Solver": "SU2"})                          # 2. stage dataset level aux data
    w.write_ordered_zone(data=[x, y, p], title="Zone 1")              # 3. write structured zones
    w.write_fe_zone(data=[...], zone_type=ZoneType.FETRIANGLE, ...)   # 4. write unstructured zones
    w.write_poly_zone(data=[...], zone_type=ZoneType.FEPOLYGON, ...)  # PLT and DAT only
    # 5. close happens automatically on context-manager exit
```

Opening with an explicit `variables` list opens the file immediately
(*eager* open); omitting it defers file creation until the first zone write,
which must then supply `variables` itself (*lazy* open). Either way, closing
(directly, or via the context manager, recommended) is required to produce a
valid, readable file.

## Order of operations

The SZL format allows for flexible order of operations, but PLT and DAT
require a sequential order of operations. However, the implementation of
writing via {func}`tecio.open` requires all three formats to write zones
sequentially. If a SZL file is desired to be written with arrays out of
sequence, {mod}`tecio.libtecio` wrapper functions must be called directly.

Zone-level aux data must be specified with the zone write methods, but
dataset aux data may be staged at any time. Upon file close, all buffered aux
data is written to the file before finalizing output.

## {class}`~tecio.TecplotWriter`

Set once at construction. These are plain attributes, not properties,
readable at any point, but reassigning one after the file header has
already been written (e.g. `title` under eager-open, once the first zone
exists) changes the Python object without changing the file, the header was
already committed.

| Attribute | Type | Description |
|---|---|---|
| `path` | `str` | Output file path |
| `title` | `str` | Dataset title |
| `variables` | `list[str] \| None` | Variable name list, `None` before the file is opened (lazy-open) |
| `file_type` | `FileType` | `FULL`, `GRID`, or `SOLUTION` |
| `current_zone` | `int` | 0-based index of the most recently written zone, `-1` before any zone is written |

`meta` (below) is a running, read-only summary of everything committed to
the file so far (title, variables, aux-item counts, and a per-zone record
in write order), useful for a quick sanity check without re-opening the
file for reading.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: writers

   TecplotWriter
```

## `write_ordered_zone`

Writes one complete IJK-ordered zone. Dimensions are inferred from the shape
of the first array (1-D → `(N, 1, 1)`, 2-D → `(I, J, 1)`, missing trailing
dimensions default to `1`).

| Parameter | Type | Default | SZL | PLT | DAT |
|---|---|---|---|---|---|
| `data` | `Sequence[ArrayLike]` | required | ✓ | ✓ | ✓ |
| `title` | `str \| None` | `None` | ✓ | ✓ | ✓ |
| `variables` | `list[str] \| None` | `None` | ✓ | ✓ | ✓ |
| `value_locations` | `Sequence[ValueLocation] \| None` | `None` | ✓ | ✓ | ✓ |
| `passive_vars` | `Collection[int \| str] \| None` | `None` | ✓ | ✓ | ✓ |
| `var_sharing` | `Mapping[int \| str, int] \| None` | `None` | ✓ | ✓ | ✓ |
| `solution_time` | `float` | `0.0` | ✓ | ✓ | ✓ |
| `strand_id` | `int` | `0` | ✓ | ✓ | ✓ |
| `aux` | `dict[str, Any] \| None` | `None` | ✓ | ✓ | ✓ |
| `datapacking` | `DataPacking \| str` | `BLOCK` | ✓ | ✓ | ✓ |
| `flush` | `bool` | `False` | ✓ | — | — |

`flush` (SZL only) writes this zone's data to a temporary file and releases
it from memory immediately, rather than holding everything until `close()`;
useful for very large datasets. Every other parameter behaves identically
across all three formats.

`var_sharing` maps a variable (by name or 0-based index) to the 0-based
index of the zone to share it from, e.g. `{"x": 0, "y": 0}` shares `x` and
`y` from zone 0. `passive_vars` takes variable names or 0-based indices,
e.g. `{"x"}` or `[3]`.

## `write_fe_zone`

Writes one complete finite-element zone. All the `write_ordered_zone` parameters
apply here too (except `datapacking`, which only means something for
ordered zones), plus:

| Parameter | Type | Default | SZL | PLT | DAT |
|---|---|---|---|---|---|
| `zone_type` | `ZoneType` | required | ✓ | ✓ | ✓ |
| `node_map` | `ArrayLike \| None` | `None` | ✓ | ✓ | ✓ |
| `con_sharing` | `int \| None` | `None` | ✓ | ✓ | ✓ |
| `face_neighbors` | `ArrayLike \| None` | `None` | ✓ | ✓ | accepted, not yet written¹ |
| `face_nbr_mode` | `FaceNeighborMode` | `LOCAL_ONE_TO_ONE` | ✓ | ✓ | accepted, not yet written¹ |
| `flush` | `bool` | `False` | ✓ | — | — |

Node and cell counts are inferred from `node_map`, or, if `node_map` is
omitted, from the zone referenced by `con_sharing`. `node_map` holds
0-based node indices into this zone's coordinate arrays. `con_sharing` is
the 0-based index of the zone to share connectivity from, or `None` to
write `node_map` directly. `face_neighbors` cell references, remote-zone
references, and the face ordinal (0..faces-per-cell - 1) are all 0-based.
Tecplot 360 itself and the data format guide's face-numbering diagrams
number faces starting at 1; `tecio` follows Python container-index
convention here instead, same as everywhere else in the Python API.

¹ **Known limitation**: DAT's `face_neighbors`/`face_nbr_mode` parameters are
accepted without error but currently do nothing, the ASCII face-neighbor
representation isn't implemented yet. Passing them silently produces a file
with no face-neighbor data, planned work, not yet scheduled.

## `write_poly_zone`

Writes one complete FEPOLYGON or FEPOLYHEDRON zone. PLT and DAT only; SZPLT
raises `NotImplementedError` (Tecplot plans SZPLT polyhedra support in a future
TecIO release). `write_fe_zone` raises `ValueError` for these zone types. The
`write_fe_zone` variable, sharing, time, strand and aux parameters apply here
too, plus:

| Parameter | Type | Default | SZL | PLT | DAT |
|---|---|---|---|---|---|
| `zone_type` | `ZoneType` | required | ✗ | ✓ | ✓ |
| `face_nodes` | `ArrayLike \| None` | `None` | ✗ | ✓ | ✓ |
| `face_node_counts` | `ArrayLike \| None` | `None` | ✗ | ✓ | ✓ |
| `face_left_elements`, `face_right_elements` | `ArrayLike \| None` | `None` | ✗ | ✓ | ✓ |
| `boundary_connection_counts` | `ArrayLike \| None` | `None` | ✗ | ✓ | ✓ |
| `boundary_connection_elements` | `ArrayLike \| None` | `None` | ✗ | ✓ | ✓ |
| `boundary_connection_zones` | `ArrayLike \| None` | `None` | ✗ | ✓ | ✓ |
| `con_sharing` | `int \| None` | `None` | ✗ | ✓ | ✓ |
| `datapacking` | `DataPacking \| str` | `BLOCK` | ✗ | BLOCK | BLOCK |

Note that all face connectivity arrays use 0-baesd node and element indices,
consistently with the node connectivity and face neighbors for classical
finite-element zones. 1D arrays for `face_left_elements`/`face_right_elements` are required to
indicate the element on each side of a face. An edge cell face with no
neighbor should use `-1`, and `-(b + 2)` to point at boundary face `b`, which
is described by the three boundary arrays.  `face_node_counts` is required for
polyhedra, but Polygon faces always have two nodes and is optional (must be
all 2 if given). Node and element counts are inferred as the highest index
plus one. If boundary faces are defined for a zone, connectivity is sharing is
disabled as the zone boundary must be uniquely defined. Tecplot requires BLOCK
packing for poly zones (even with only nodal variables) so this package will
raise `ValueError` for `datapacking="POINT"`.

## Format-specific classes

Constructor and a handful of format quirks; everything else is the shared
interface above.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :toctree: writers

   TecplotSzlWriter
   TecplotPltWriter
   TecplotDatWriter
```

## Precision

Each writer's `precision` constructor argument means something different,
this is the one constructor parameter worth understanding up front rather
than discovering by trial and error.

| | SZL | PLT | DAT |
|---|---|---|---|
| Default | `None` (per-variable, automatic) | `DataType.DOUBLE` | `DataType.FLOAT` |
| Scope | Optional whole-file override; without it, each variable's on-disk type is inferred from its own array's dtype | Whole file, unconditionally; the classic API has no per-variable type at all, even integers are stored as the chosen float type | Whole file for the *float-vs-double* choice and the printed significant-digit count; integer-inferred variables always keep their own declared type (`DT=` in the header) regardless of `precision` |

DAT is the one format where an integer variable's memory footprint on read
genuinely depends on what was written, not on `precision`, e.g. a `count`
array written alongside `float64` coordinates under `precision=DataType.DOUBLE`
still declares (and reads back) as `LONGINT`, only the printed digit count
for the *floating-point* variables changes.
