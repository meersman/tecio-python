(installation)=
# Installation

The {mod}`tecio` package was written and tested for Linux and maxOS operating
systems. In principle, any system that is capable of running Tecplot with a
modern Python install should be able to run {mod}`tecio`. 

The recommended installation method is via PyPI:

```bash
pip install tecio-python
```

This will also install or upgrade the required Python dependency,
[NumPy](https://numpy.org).

:::{note}
On systems where you do not have elevated privileges add the `--user` flag so
the package is installed into your home directory instead of the system-wide
site-packages:

    pip install --user tecio-python

The installed scripts (`tecdump`, `tecstats`, etc.) will be placed in
`~/.local/bin` on Linux and macOS. Make sure that directory is on your `PATH`.
:::

Alternatively, install a specific version directly from the wheel distributed
with each [GitHub release](https://github.com/meersman/tecio/releases):

```bash
pip install tecio-<version>-py3-none-any.whl
```

If you have cloned the repository and prefer to install from source:

```bash
# From the repository root
pip install .
```

For development (editable install with all optional tooling):

```bash
pip install -e ".[dev]"
```

---

(dependencies)=
## Software Dependencies

### Required

- **Python** 3.10 or later
- **NumPy** — installed automatically by `pip`
- **Tecplot 360 EX** (2024 R1 or newer) — provides the TecIO shared library
  (`libtecio.so` on Linux, `libtecio.dylib` on macOS, `tecio.dll` on
  Windows). An active Tecplot *license* is **not** required at runtime;
  only the installed software distribution is needed.

:::{note}
The `FEMIXED` zone type was added to TecIO released with Tecplot 360 EX 2024
R1. {mod}`tecio` is compatible with older versions of TecIO, but `FEMIXED`
zones are not supported if an older version of the TecIO shared object binary
is used.
:::

### Optional

The demos under `demos/` use additional packages that are not installed
automatically:

- [SciPy](https://scipy.org/) — spectral solver and interpolation demos
- [tqdm](https://tqdm.github.io/) — progress bars during long time loops

Install them together with:

```bash
pip install scipy tqdm
```

The [IPython](https://ipython.org/) interactive shell is recommended for users
who want to use the API to read or write Tecplot data interactively.

---

(locating-tecio)=
## Locating the TecIO Shared Library

On import, `tecio` searches for the TecIO shared library by locating the
Tecplot 360 executable on `PATH` and resolving the library path relative to
it.  This works automatically for standard Tecplot installations.

If the automatic search fails — for example, because Tecplot is not on `PATH`,
multiple versions are installed, or you are working on an HPC system where
Tecplot is loaded as a module — set the `TECIO_LIB` environment variable to
the full path of the library:

```bash
# Linux
export TECIO_LIB=/opt/tecplot/360ex_2025r1/bin/libtecio.so

# macOS
export TECIO_LIB="/Applications/Tecplot 360 EX 2025 R1.app/Contents/Frameworks/libtecio.dylib"

# Windows (PowerShell)
$env:TECIO_LIB = "C:\Program Files\Tecplot\Tec360EX 2025 R1\bin\tecio.dll"
```

Add the export to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) or to
your project's `.env` file to make the setting permanent.

To verify which library `tecio` has resolved at runtime:

```python
from tecio import utils

print(utils.get_tecio_lib())
```
