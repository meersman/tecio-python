"""Shared pytest configuration and fixtures for tecio tests.

Add ``--keep-files`` to any pytest run to keep output files in
``tests/output/`` for manual inspection in Tecplot 360::

    $ pytest tests/ -v --keep-files
    $ pytest tests/test_szl_write.py::TestWriteFEZone::test_write_fe_tet -v --keep-files
"""

from __future__ import annotations

from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# --keep-files flag
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--keep-files`` command-line option."""
    parser.addoption(
        "--keep-files",
        action="store_true",
        default=False,
        help=(
            "Write output files to tests/output/<test-name>/ instead of a "
            "temporary directory. Files persist after the run so they can "
            "be opened in Tecplot 360 for visual verification."
        ),
    )


@pytest.fixture
def output_path(request: pytest.FixtureRequest, tmp_path: Path):
    """Return a factory function that resolves output file paths.

    Without ``--keep-files`` (default): files go to pytest's temporary
    directory and are deleted after the run.

    With ``--keep-files``: files go to ``tests/output/<test-name>/`` and
    persist after the run.

    Usage in a test::

        def test_something(self, output_path):
            path = output_path("my_file.szplt")
            with tecio.open(str(path), "w") as w:
                ...
    """
    if request.config.getoption("--keep-files"):
        # Build a stable, filesystem-safe name from the full test node id.
        name = request.node.nodeid
        for ch in r"[]/:\ ":
            name = name.replace(ch, "_")
        out_dir = _TESTS_DIR / "output" / name
        out_dir.mkdir(parents=True, exist_ok=True)

        def resolve(filename: str) -> Path:
            return out_dir / filename

    else:

        def resolve(filename: str) -> Path:
            return tmp_path / filename

    return resolve


# ---------------------------------------------------------------------------
# Shared Onera test-file paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Return the ``tests/`` directory as a Path."""
    return _TESTS_DIR


@pytest.fixture(scope="session")
def onera_szplt() -> Path:
    """Return the Onera SZL test file path."""
    return _TESTS_DIR / "Onera.szplt"


@pytest.fixture(scope="session", params=["szplt", "plt", "dat"])
def onera_path(request: pytest.FixtureRequest) -> Path:
    """Parametrised fixture: Onera file in each supported format.

    Returns:
        Path of the Onera file in the current format.
    """
    return _TESTS_DIR / f"Onera.{request.param}"
