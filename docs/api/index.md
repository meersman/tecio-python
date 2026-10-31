# API Reference

{mod}`tecio` is designed around a single entry point, {func}`tecio.open`,
which opens a Tecplot file for reading, writing, or appending and returns the
appropriate reader or writer based on the file extension. In most cases you
will not need to instantiate {class}`~tecio.TecplotSzlReader`,
{class}`~tecio.TecplotSzlWriter`, or any other class directly;
{func}`tecio.open` selects and returns the correct one.


<div style="display: flex; gap: 20px;">

<div style="flex: 1;">

:::{table}

| Mode | Description |
|------|-------------|
| `"r"` | Read an existing file |
| `"w"` | Write a new file |
| `"x"` | Write, failing if the file already exists |
| `"a"` | Append new zones to an existing file |
| `"a+"` | Append and read in the same session |
:::

</div>

<div style="flex: 1;">

:::{table}

| Extension | Format |
|-----------|--------|
| `.szplt` | Tecplot SZL binary (subzone-loadable) |
| `.plt`, `.bin`, `.bbin`, `.bbin8` | Tecplot PLT binary |
| `.dat`, `.tec` | Tecplot ASCII |
:::

</div>

</div>

---

## Indexing

From the [Tecplot Data Format Guide](https://tecplot.azureedge.net/products/360/current/360-data-format.html#reading-szl/getting-started/indexing),

> All indexing in Tecplot data files (including zones and variables) is
  1-based, so the last index is exactly equal to (not one less than) the count
  of the element type. An exception is nodes in finite-element zones, which
  can be specified to be zero-based. C and C++ programmers in particular
  should be wary of off-by-one errors when dealing with Tecplot data.

`tecio`'s Python API follows standard Python convention everywhere instead:
every zone, variable, node, and cell index it accepts or reports is 0-based,
the same as indexing a `list` or a NumPy array. This applies uniformly, from
container indexing to `zone_index`/`var_index` to every function or method
parameter that names a zone or variable.

Container indexing is 0-based:

```python
r.zones[0]  # the FIRST zone
r.zones[0].variables[0]  # the FIRST variable in that zone
```

`zone_index` reports that same 0-based position:

```python
r.zones[0].zone_index  # == 0
r.zones[2].zone_index  # == 2
```

For read variables, `var_index` reports the 0-based position within its
zone, and `zone_index` reports the 0-based index of the parent zone:

```python
r.zones[0].variables[0].var_index  # == 0
r.zones[2].variables[0].zone_index  # == 2
```

Data access method `get_array` also uses 0-based indexing:

```python
r.zones[0].get_array(0)  # data array for FIRST variable
r.zones[0].get_array([1, 3, 5])  # tuple of data arrays for 2nd, 4th and 6th variables
```

Auxiliary data access follows the same rule. `get_var_auxdata` and
`get_zone_auxdata` take a 0-based index, same as `zone_index` and
`var_index`:

```python
r.get_var_auxdata(0)  # the FIRST variable's aux data
r.get_zone_auxdata(0)  # the FIRST zone's aux data
```

Two places keep Tecplot's native 1-based numbering on purpose, since they
hand values to (or read them from) something other than this Python API:
the raw {mod}`tecio.libtecio` C function wrappers, which pass zone/variable/
node arguments straight through to TecIO, and the `tec2mat` CLI, which
converts every index to 1-based on the way into its `.mat` output so the
same values work directly in MATLAB. The *text* of ASCII `.dat`/`.tec` files
is also 1-based, matching how Tecplot 360 itself reads and writes that
format, but values `tecio` parses back out of a `.dat` file follow the same
0-based convention as every other format.

---

## Core Classes

These classes define the interface every reader and writer shares, regardless
of which of the three file formats produced or is producing the data. See
{doc}`readers` and {doc}`writers` for the full property/method reference.

**Reading**

| Class | Description |
|---|---|
| {class}`~tecio.TecplotReader` | One open file: title, variables, zones, dataset-level aux data |
| {class}`~tecio.TecplotZoneReader` | Shared zone interface: title, type, solution time, strand ID, variables, aux data |
| {class}`~tecio.TecplotOrderedZoneReader` | Adds `I`/`J`/`K`/`dimensions` for IJK-ordered zones |
| {class}`~tecio.TecplotFEZoneReader` | Adds `node_map`/`nodes_per_cell`/`shared_connectivity` for finite-element zones |
| {class}`~tecio.TecplotPolyZoneReader` | Adds the face map (`face_nodes`, `face_left`, `face_right`, boundary connections) for polygon and polyhedron zones |
| {class}`~tecio.TecplotVariableReader` | One variable's metadata and data within a zone |
| {class}`~tecio.TecplotAuxDataReader` | Read-only, dict-like view of one auxiliary-data mapping |

**Writing**

| Class | Description |
|---|---|
| {class}`~tecio.TecplotWriter` | Shared writer interface: construction, aux-data staging, zone writing, closing |

{class}`~tecio.TecplotZoneReader` returns either
{class}`~tecio.TecplotOrderedZoneReader`,
{class}`~tecio.TecplotFEZoneReader` or
{class}`~tecio.TecplotPolyZoneReader` for structured, unstructured and
polygon/polyhedron data respectivly as these zone types have properties that
are not shared (e.g. `dimensions` vs. `node_map` vs. `face_nodes`). Accessing a zone property no included in
the zone type raises `AttributeError`. To check, use `isinstance(zone,
TecplotOrderedZoneReader)` (or the FE equivalent) or zone reader object
property `zone_type` for the more in depth identifyer
{class}`~tecio.ZoneType`.

---

## Format-Specific Classes

{func}`tecio.open` constructs one of these six classes for you, based on the
file extension. Construct one directly only if you want to bypass the
extension-based dispatch, e.g. to force a specific format regardless of file
name.

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   TecplotSzlReader
   TecplotPltReader
   TecplotDatReader
   TecplotSzlWriter
   TecplotPltWriter
   TecplotDatWriter
```

See {doc}`readers` and {doc}`writers` for each class's full documentation,
including per-format memory-model notes and parameter comparison tables.
{class}`~tecio.TecplotPltReader` also reads several older PLT binary versions
(v75, v107, v111), see {doc}`legacy_plt_versions`.

---

## Accessing Variable Data

Every reader's `zones` and `variables` properties return one of two
format-agnostic container types rather than a plain `list`. Both support the
same indexing, iteration, and `len()` you would expect from a list, plus
name-based and slice-based access:

```python
with tecio.open("flow.szplt") as r:
    r.zones[0]  # -> TecplotOrderedZoneReader, TecplotFEZoneReader, or TecplotPolyZoneReader
    r.zones[1:4]  # -> ZoneList (sub-range, same kind)
    r.zones[0].variables  # -> VariableList
    r.zones[0].variables["x"]  # -> TecplotVariableReader, by exact name
    r.zones[0].variables[2]  # -> TecplotVariableReader, by 0-based index
```

Indexing a {class}`~tecio.ZoneList` or {class}`~tecio.VariableList` always
returns an element or a sub-collection of the *same* kind — never a raw
array. To pull the underlying NumPy data for one or more variables in a
single zone, use `TecplotZoneReader.get_array`, shared by every zone type
and every format:

```python
p = r.zones[0].get_array("p")  # ndarray | None
p = r.zones[0].get_array(2)  # by 0-based index
x, y, z = r.zones[0].get_array(["x", "y", "z"])  # tuple, for unpacking
```

A single key (index or name) returns one array; a list of names returns a
tuple of arrays in the order given, suitable for unpacking. There is
deliberately no cross-zone array accessor — to pull one variable across many
zones (e.g. a transient sequence), iterate explicitly so the outer axis stays
in your code, and stack only when you know the shapes match:

```python
seq = [z.get_array("p") for z in r.zones]  # list[ndarray | None]
stack = np.stack(seq)  # only if every zone matches
```

`get_array` returns `None` for a passive or shared variable, and raises
`KeyError` for an unknown name or `IndexError` for an out-of-range index.

| Class | Returned by | Description |
|---|---|---|
| {class}`~tecio.ZoneList` | `TecplotReader.zones` | Sequence of zones; integer index, slice, or iterate |
| {class}`~tecio.VariableList` | `TecplotZoneReader.variables` | Sequence of variables; index by position or exact name |

See {doc}`readers` for the full container API.

---

## Meaningful Constants

Zone types, data types, value locations, and the rest of Tecplot's
enumerated constants are plain Python enums with no dependency on the C
library, importable directly from {mod}`tecio` (e.g. `tecio.ZoneType`).

```{eval-rst}
.. currentmodule:: tecio

.. autosummary::
   :nosignatures:

   FileFormat
   FileType
   ZoneType
   DataType
   ValueLocation
   FaceNeighborMode
   FeCellShape
   DataPacking
   VarStatus
   Boolean
   Debug
```

See {doc}`constants` for each constant's full member listing.

---

## C Function Wrappers

{mod}`tecio.libtecio` provides the direct Python wrapper for every TecIO C
function tecio uses internally, kept public so the library can be called
directly, if desired. Importing it never raises, even with no Tecplot
installation at all, or an older one missing a newer function, see
{doc}`libtecio` for how that degradation works.

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   TecioError
   TecioUnavailableError
   TecioAvailabilityWarning
```

**SZL Read Functions**

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tec_file_reader_open
   tec_file_reader_close
   tec_file_get_type
   tec_data_set_get_title
   tec_data_set_get_num_vars
   tec_data_set_get_num_zones
   tec_zone_get_ijk
   tec_zone_get_title
   tec_zone_get_type
   tec_zone_is_enabled
   tec_zone_get_solution_time
   tec_zone_get_strand_id
   is_64bit
   tec_zone_node_map_get_64
   tec_zone_node_map_get
   tec_var_get_name
   tec_var_is_enabled
   tec_zone_var_get_type
   tec_zone_var_get_value_location
   tec_zone_var_is_passive
   tec_zone_var_get_shared_zone
   tec_zone_var_get_num_values
   tec_zone_var_get_float_values
   tec_zone_var_get_double_values
   tec_zone_var_get_int32_values
   tec_zone_var_get_int16_values
   tec_zone_var_get_uint8_values
   tec_data_set_aux_data_get_num_items
   tec_data_set_aux_data_get_item
   tec_var_aux_data_get_num_items
   tec_var_aux_data_get_item
   tec_zone_aux_data_get_num_items
   tec_zone_aux_data_get_item
```

**SZL Write Functions**

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tec_file_writer_open
   tec_file_writer_close
   tec_file_writer_flush
   tec_zone_create_ijk
   tec_zone_create_fe
   tec_zone_set_unsteady_options
   tec_zone_var_write_double_values
   tec_zone_var_write_float_values
   tec_zone_var_write_int32_values
   tec_zone_var_write_int16_values
   tec_zone_var_write_uint8_values
   tec_zone_node_map_write32
   tec_zone_node_map_write64
   tec_zone_face_nbr_write_connections32
   tec_zone_face_nbr_write_connections64
   tec_data_set_add_aux_data
   tec_var_add_aux_data
   tec_zone_add_aux_data
```

**Classic API Functions**

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tecini142
   tecend142
   tecflush142
   tecfil142
   tecforeign142
   teczne142
   tecpolyzne142
   tecznefemixed142
   tecdat142
   tecnode142
   tecface142
   tecpolyface142
   tecpolybconn142
   tecauxstr142
   tecvauxstr142
   teczauxstr142
   tecusr142
```

See {doc}`libtecio` for the full module page, and {doc}`libtecio_szl_read`,
{doc}`libtecio_szl_write`, {doc}`libtecio_classic` for each function's full
signature and documentation.

---

## Append Handles

Returned by {func}`tecio.open` when using `'a'` or `'a+'` mode.

| Class | Mode | Description |
|---|---|---|
| {class}`~tecio.AppendWrite` | `'a'` | Append zones to an existing SZL file |
| {class}`~tecio.AppendReadWrite` | `'a+'` | Append zones and read existing zones |

```{toctree}
:hidden:

open
readers
writers
constants
libtecio
legacy_plt_versions
```
