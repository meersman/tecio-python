# Configuration file for the Sphinx documentation builder.
# ruff: noqa D100 D103

import shutil
import sys
import tomllib
from pathlib import Path

# -- Project information -----------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

with PYPROJECT.open("rb") as f:
    data = tomllib.load(f)

project_info = data["project"]
project = project_info["name"]
release = project_info["version"]
copyright = "2026, John Meesman"
author = "John Meersman"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.matlab",
    "myst_parser",
    "paraview_stubs",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Napoleon settings (Google-style docstrings) -----------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# napoleon_include_special_with_doc = True
napoleon_include_special_with_doc = False

napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_param = False
napoleon_use_rtype = False

# -- Autodoc settings --------------------------------------------------------

autodoc_default_options = {
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_typehints = "none"
# autodoc_class_signature = "separated"
autodoc_class_signature = "mixed"

# Add the ParaView plugins to the path so autodoc can import them. The local
# paraview_stubs extension stands in for ParaView and VTK, which are not installed for
# the docs build.
sys.path.insert(0, str(ROOT / "plugins" / "paraview"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

matlab_src_dir = str(ROOT / "plugins")

# -- Autosummary settings ----------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- MyST parser -------------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "tecio"
html_static_path = ["_static"]
html_last_updated_fmt = "%m-%d-%Y"

html_theme_options = {
    "repository_url": "https://github.com/meersman/tecio",
    "use_repository_button": True,
    "repository_branch": "main",
    "use_download_button": True,
    "show_toc_level": 2,
    "toc_title": "On this page",
    "navigation_with_keys": True,
    "back_to_top_button": True,
    "pygments_light_style": "github-light",
    "pygments_dark_style": "paraiso-dark",
}

html_extra_path = ["_static/googlec97c014dabffcdf6.html"]
