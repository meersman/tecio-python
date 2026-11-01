# tecio

A Python interface for reading and writing Tecplot data files.

This package wraps Tecplot's TecIO C library for reading and writing binary
SZL (`.szplt`) and PLT (`.plt`, `.bin`) formats as well as providing
read/write utilities for ASCII DAT (`.dat`, `.tec`) files.

The Tecplot data format is organised around *zones*, a flexible object that
can represent a structured or unstructured block, subdomain, or boundary
surface of a CFD solution, a single time step in a transient dataset, or a
line or scatter point in a plot. Zone data may be *structured* (IJK-ordered
matrix) or *unstructured* (nodes and cells with an explicit connectivity
map). A file may contain both grid and solution data, or just one or the other
via the GRID and SOLUTION file types.

All zones in a dataset share the same variable list, though individual
variables may be marked *passive* (no data for that zone) or *shared* from
another zone to avoid storing redundant data, commonly used to share grid
coordinates across time steps. Auxiliary data can be attached at the dataset,
variable, or zone level as free-form key-value strings. Special `Common.*`
auxiliary data keys are recognised by Tecplot 360 EX on load and can be used
to automatically configure axis variables, velocity vectors, the default
contour variable, and more.

The SZL (`.szplt`) format additionally uses a proprietary node-map compression
algorithm that produces significantly smaller files compared to the classic
PLT format, particularly for large unstructured meshes.

> The TecIO C library is proprietary software distributed by Tecplot,
  Inc. under a permissive license agreement. This package is an independent
  utility and is not affiliated with or endorsed by Tecplot, Inc. Use of
  `tecio` requires an existing Tecplot installation to provide the shared
  library, but does not require an active Tecplot license.

---

## Contents

1. **[Installation](installation.md)**
   - {ref}`Software Dependencies <dependencies>`
   - {ref}`Locating the TecIO Shared Library <locating-tecio>`
   
2. **[Quickstart](quickstart.md)**
   - {ref}`Examples <qs-examples>`
   - {ref}`Indexing Conventions <qs-index-convention>`
   - {ref}`Getting Help <qs-getting-help>`

3. **[API Reference](api/index.md)**
   - [`tecio.open`](api/index.md) — open a file for reading, writing, or appending
     - [`AppendWrite`](api/append_write.md) — append zones to an existing file
       (`mode='a'`)
     - [`AppendReadWrite`](api/append_read_write.md) — append and read in the
       same session (`mode='a+'`)
   - **[Readers](api/readers.md)** — the shared reader interface,
       `ZoneList`/`VariableList`, and each format's reader class
   - **[Writers](api/writers.md)** — the shared writer interface,
       order-of-operations, and each format's writer class
   - **[Constants](api/constants.md)** — `ZoneType`, `DataType`, and the rest
       of Tecplot's enumerated constants
   - **[C Function Wrappers](api/libtecio.md)** — low-level TecIO C library bindings
     - [SZL Read Functions](api/libtecio_szl_read.md)
     - [SZL Write Functions](api/libtecio_szl_write.md)
     - [Classic API Functions](api/libtecio_classic.md)

4. **[Console Scripts](api/cli.md)** — command-line tools

5. **Third Party Plugins**
   - [ParaView](plugins/paraview.md) — use `tecio-python` to natively read and
     write Tecplot files in ParaView.
   - [MATLAB](plugins/matlab) — complementary functions to manipulate and plot
     Tecplot data converted to MATLAB with {mod}`~tecio.cli.tec2mat`.

6. **Demos**
   - [Lorenz Attractor Animation](_demos/lorenz/lorenz.md)

---

```{toctree}
:hidden:
:maxdepth: 2
:caption: User Guide

installation
quickstart
api/index
api/cli
changelog
genindex
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Plugins

plugins/paraview
plugins/matlab
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Demos

_demos/lorenz/lorenz
```
