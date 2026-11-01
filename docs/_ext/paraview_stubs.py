"""Sphinx extension that stands in for ParaView and VTK while the docs are built.

The plugins in ``plugins/paraview`` import ``paraview`` and ``vtkmodules`` at the top of
the module and decorate their classes and methods with ``smproxy``, ``smproperty``,
``smdomain`` and ``smhint``. Neither package is installed when the docs are built, and
``autodoc_mock_imports`` cannot be used because a mocked decorator replaces the class
or method it decorates, which hides its docstring from autodoc.

The stand-ins installed here keep the decorated objects unchanged: every decorator
returns the object it is given, every VTK name is an empty class, and
``VTKPythonAlgorithmBase`` is a plain base class. They are only installed if the real
packages cannot be imported, so a docs build in a ParaView Python still uses ParaView.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from sphinx.application import Sphinx

_DECORATOR_MODULES = ("smdomain", "smhint", "smproperty", "smproxy")


class _Decorators:
    """Namespace whose attributes are decorator factories that change nothing."""

    def __getattr__(self, name: str) -> Callable[..., Callable[[Any], Any]]:
        """Return a decorator factory for the attribute *name*.

        Returns:
            A function that accepts any arguments and returns a decorator that returns
            the decorated object.

        Raises:
            AttributeError: If *name* is a dunder name.
        """
        if name.startswith("__"):
            raise AttributeError(name)

        def factory(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
            return lambda obj: obj

        return factory


class _StubModule(types.ModuleType):
    """Module whose missing attributes are empty classes, like ``vtkDataSet``."""

    def __getattr__(self, name: str) -> Any:
        """Create an empty class called *name* on first access.

        Returns:
            The new class, which is also stored on the module.

        Raises:
            AttributeError: If *name* is a dunder name.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (), {})
        setattr(self, name, value)
        return value


def _install() -> None:
    """Put the stand-in modules in ``sys.modules`` unless the real ones import."""
    if importlib.util.find_spec("paraview") is not None:
        return

    names = (
        "paraview",
        "paraview.util",
        "vtkmodules",
        "vtkmodules.util",
        "vtkmodules.util.numpy_support",
        "vtkmodules.util.vtkAlgorithm",
        "vtkmodules.vtkCommonCore",
        "vtkmodules.vtkCommonDataModel",
        "vtkmodules.vtkFiltersCore",
    )
    for name in names:
        sys.modules[name] = _StubModule(name)

    algorithm = _StubModule("paraview.util.vtkAlgorithm")
    for name in _DECORATOR_MODULES:
        setattr(algorithm, name, _Decorators())
    sys.modules["paraview.util.vtkAlgorithm"] = algorithm


_install()


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension, which only needs to be imported.

    Returns:
        Extension metadata for Sphinx.
    """
    return {"parallel_read_safe": True, "parallel_write_safe": True}
