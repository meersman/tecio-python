# Contributing

Thanks for your interest in contributing to tecio-python! This is a
single-author side project, so while contributions are welcome, I may take
some time to review PRs.

## How to contribute

1. **Fork** the repository on GitHub
2. **Clone** your fork and create a branch for your changes
3. **Commit** your changes with clear, descriptive commit messages
4. **Push** your branch and open a **pull request** against `main`

## Code quality

The linting rules and unit test piplines can be run locally with

```bash
make check   # runs ruff format, ruff lint, and ty typecheck
make test    # runs the full test suite
```

- **Formatting** — `ruff format` (line length 88, PEP 8 conventions)
- **Linting** — `ruff check` (pycodestyle, pydocstyle, pyflakes, isort)
- **Type checking** — `ty check` (Python 3.10+ type hints required)
- **Tests** — `pytest` (new functionality should include tests)

These same checks run automatically on every pull request via GitHub Actions.

## Development setup

```bash
git clone https://github.com/meersman/tecio-python.git
cd tecio-python
make venv
make install
```

You will also need a Tecplot 360 installation or the TecIO shared library
available. Set the `TECIO_LIB` environment variable to point to the library:

```bash
export TECIO_LIB=/path/to/libtecio.so   # Linux
export TECIO_LIB=/path/to/libtecio.dylib  # macOS
```
