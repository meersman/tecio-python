# ParaView Plugins

`tecio` reads Tecplot data from any of its file formats into NumPy arrays, which
makes it straightforward to hand that data to VTK objects and from there to
ParaView. The plugins in the `plugins/paraview` folder of the repository do this
as a ParaView reader and writer for Tecplot files.

ParaView already has readers for binary PLT and ASCII Tecplot files. However, it
has no support for the Tecplot SZL format, and its Tecplot reader does not
properly read `STRANDID` in the zone headers of DAT files. These plugins also
carry Tecplot auxiliary data into VTK field data, and read and write polygon and
polyhedron zones.

They are intended for users who have access to Tecplot (through a university or
other institution, for example) but prefer to visualize their data in ParaView.
Tecplot files are a common output of CFD solvers and meshing software, and the
format stores complex multi-block and time-dependent field data efficiently.

| Plugin | File | ParaView name | Purpose |
|---|---|---|---|
| Reader | `TecplotTecioReader.py` | Tecplot 360 Reader (TecIO) | Opens `.szplt`, `.plt`, `.bin`, `.dat` and `.tec` files as a multi-block dataset with one block per zone |
| Writer | `TecplotTecioWriter.py` | Tecplot 360 Writer (TecIO) | Saves a dataset or multi-block dataset to one of the same file types, chosen by the file extension |

## Requirements

- A ParaView with Python support.
- The `tecio` package must be importable from the same Python interpreter that
  ParaView uses to run the plugins.
- `tecio` must be able to find the TecIO shared library from a Tecplot
  installation, see {ref}`Locating the TecIO Shared Library <locating-tecio>`.

## Loading a plugin

1. In ParaView, open **Tools > Manage Plugins...**.
2. Click **Load New...** and select `TecplotTecioReader.py` or
   `TecplotTecioWriter.py`.
3. Expand the plugin in the list and tick **Auto Load** to load it each time
   ParaView starts.

Once loaded, the reader handles the supported extensions in **File > Open** and
the writer is offered under **File > Save Data**.

---

## Reader

```{eval-rst}
.. automodule:: TecplotTecioReader
   :no-members:
```

### ParaView properties

The reader exposes these properties in the Properties panel. Each one is backed
by a method of the plugin class.

| Property | Method |
|---|---|
| `FileName` | {meth}`~TecplotTecioReader.TecplotReader.SetFileName` |
| `Arrays` | {meth}`~TecplotTecioReader.TecplotReader.GetDataArraySelection` |
| `Zones` | {meth}`~TecplotTecioReader.TecplotReader.GetZoneSelection` |
| `XArray`, `YArray`, `ZArray` | {meth}`~TecplotTecioReader.TecplotReader.SetXArray`, {meth}`~TecplotTecioReader.TecplotReader.SetYArray`, {meth}`~TecplotTecioReader.TecplotReader.SetZArray` |
| `UArray`, `VArray`, `WArray` | {meth}`~TecplotTecioReader.TecplotReader.SetUArray`, {meth}`~TecplotTecioReader.TecplotReader.SetVArray`, {meth}`~TecplotTecioReader.TecplotReader.SetWArray` |
| `VariableInfo` | {meth}`~TecplotTecioReader.TecplotReader.GetVariableInfo` |
| `TimestepValues` | {meth}`~TecplotTecioReader.TecplotReader.GetTimestepValues` |

### Class reference

```{eval-rst}
.. currentmodule:: TecplotTecioReader

.. autosummary::
   :toctree: reader
   :nosignatures:

   TecplotReader
```

---

## Writer

```{eval-rst}
.. automodule:: TecplotTecioWriter
   :no-members:
```

### ParaView properties

| Property | Method |
|---|---|
| `FileName` | {meth}`~TecplotTecioWriter.TecplotWriter.SetFileName` |

### Class reference

```{eval-rst}
.. currentmodule:: TecplotTecioWriter

.. autosummary::
   :toctree: writer
   :nosignatures:

   TecplotWriter
```
