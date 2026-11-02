# tecio

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/meersman/tecio-python/actions/workflows/ci.yml/badge.svg)](https://github.com/meersman/tecio/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/meersman/tecio-python/branch/main/graph/badge.svg)](https://codecov.io/gh/meersman/tecio-python)
[![Docs](https://github.com/meersman/tecio-python/actions/workflows/docs.yml/badge.svg)](https://meersman.github.io/tecio-python)
[![PyPI](https://img.shields.io/pypi/v/tecio-python)](https://pypi.org/project/tecio-python)
[![Python](https://img.shields.io/pypi/pyversions/tecio-python)](https://pypi.org/project/tecio-python)

Python interface for reading and writing Tecplot data files.

## Overview

`tecio` wraps Tecplot's TecIO C library to provide a Python API for reading
and writing Tecplot binary formats (`.szplt`, `.plt`) and ASCII (`.dat`). It
also includes command-line tools for common file operations such as converting
formats, extracting zones, and computing statistics.

**Requirements:** Python 3.10+, NumPy, Tecplot 360 (or the standalone TecIO library)

---

## Installation

```bash
pip install tecio-python
```

The TecIO shared library (`libtecio.so` / `libtecio.dylib`) is not bundled
with this package and must be provided separately via a Tecplot 360
installation or the standalone TecIO library. Set the `TECIO_LIB` environment
variable to point to the library if it is not on your system path:

```bash
export TECIO_LIB=/path/to/libtecio.so     # Linux
export TECIO_LIB=/path/to/libtecio.dylib  # macOS
```

## Quick start

Writing a file:

```python
import tecio
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

with tecio.open("sine.szplt", "w") as tec:
    tec.write_ordered_zone(data=[x, y], variables=["x", "y"])
```

Reading a file:

```python
with tecio.open("sine.szplt", "r") as tec:
    print(tec.variables)  # ['x', 'y']
    x = tec.zones[0].variables[0].values
    y = tec.zones[0].variables[1].values
    # or
    x, y = tec.zones[0].get_array(["x", "y"])
```

## API

All file access goes through `tecio.open`, which selects the correct
format handler from the file extension and returns a context manager:

| Mode | Description |
|------|-------------|
| `"r"` | Read an existing file |
| `"w"` | Write a new file |
| `"x"` | Write, failing if the file already exists |
| `"a"` | Append new zones to an existing file |
| `"a+"` | Append and read in the same session |

Supported formats:

| Extension | Format |
|-----------|--------|
| `.szplt` | Tecplot SZL binary (subzone-loadable) |
| `.plt` / `.bin` | Tecplot PLT binary |
| `.dat` / `.tec` | Tecplot ASCII |

---

## Contributing

Contributions are welcome (see [CONTRIBUTING.md](CONTRIBUTING.md) for details).

---

## References

- [Tecplot data format specification](https://tecplot.azureedge.net/products/360/current/360-data-format.html)
- [TecIO library](https://tecplot.com/products/tecio-library/)
- [Full documentation](https://meersman.github.io/tecio-python)

---

## Disclaimer

This project is an independent open source library and is not affiliated with,
endorsed by, or supported by Tecplot, Inc. in any way. Tecplot and TecIO are
trademarks of Tecplot, Inc. Use of the TecIO library requires separatly
aquiring the source code or compiled binaries from Tecplot, Inc.
