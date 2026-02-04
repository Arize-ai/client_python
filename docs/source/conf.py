"""Sphinx configuration for Arize Python SDK API documentation."""

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# Add the src directory to the Python path so autodoc can import the modules
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# -- Project information -----------------------------------------------------

project = "Arize Python SDK"
copyright = "2025, Arize AI"
author = "Arize AI"

# Read version from arize/version.py
version_file = Path(__file__).parents[2] / "src/arize/version.py"
exec(version_file.read_text())
__version__: str  # Defined by exec() above
version = release = __version__

# -- Versioning --------------------------------------------------------------

# Determine version for the version switcher dropdown
version_match = os.environ.get("READTHEDOCS_VERSION")

# Handle different build contexts:
# - Local builds: Use "latest" as default
# - PR builds on RTD: Use "latest" (READTHEDOCS_VERSION is a number)
# - "stable" tag on RTD: Use "latest"
# - Version-specific builds: Use the READTHEDOCS_VERSION as-is
if not version_match or version_match.isdigit() or version_match == "stable":
    version_match = "latest"

# URL to the switcher.json file
# For local builds, use the local file; for ReadTheDocs, use the remote URL
if version_match == "latest" and not os.environ.get("READTHEDOCS"):
    # Local build - use relative path to local switcher.json
    json_url = "_static/switcher.json"
else:
    # ReadTheDocs build - use remote URL
    json_url = "https://arize-client-python.readthedocs.io/en/latest/_static/switcher.json"

# -- General configuration ---------------------------------------------------

source_suffix = [".rst", ".md"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST parser settings (Markdown)
myst_enable_extensions = [
    "colon_fence",  # ::: fences for directives
    "deflist",  # Definition lists
    "substitution",  # Variable substitutions
    "tasklist",  # Task lists [ ]
]
myst_heading_anchors = 2

# Copybutton settings - strip Python prompts when copying
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings:
# -- In Sphinx autodoc, boolean flags like "show-inheritance" can't be set to False
#   to disable them. Sphinx only checks if the key exists in the dictionary - if
#   it does, the feature is enabled regardless of the value.
autodoc_default_options = {
    # Include all members (methods, properties, attributes) by default
    "members": True,
    # Don't include private members (prefixed with single underscore _)
    "private-members": False,
    # Don't include special/dunder members like __str__, __repr__ (empty string = none)
    "special-members": "",
    # Don't include members that lack docstrings
    "undoc-members": False,
    # Don't include members inherited from base classes (prevents duplication)
    "inherited-members": False,
    # Show base classes in class signatures (e.g., "class Foo(Bar):")
    "show-inheritance": True,
    # Globally exclude these specific members from all classes
    "exclude-members": "__init__, __weakref__",
}
# Where to display type hints: "signature" (in function signature), "description" (in Args section), or "both"
autodoc_typehints = "both"  # Show types in parameter descriptions for cleaner signatures
# What to include in class docs: "class" (only class docstring), "init" (only __init__), "both" (concatenate both)
autoclass_content = "both"  # Concatenate class docstring and __init__ docstring
# Don't add "()" after function/method names in cross-references (cleaner look)
add_function_parentheses = False
# Don't prepend full module path to class/function names (e.g., "ArizeClient" not "arize.client.ArizeClient")
add_module_names = False
# Show actual default values in signatures instead of ellipsis (e.g., "timeout=30" not "timeout=...")
autodoc_preserve_defaults = True
# Where to add type information in descriptions: "all" (all params), "documented" (only params in docstring), "documented_params" (only in Args section)
autodoc_typehints_description_target = "all"
# Format for type hints in descriptions
autodoc_typehints_format = "short"

# Autosummary
autosummary_generate = True

# Templates
templates_path = ["_templates"]

# Exclude patterns
exclude_patterns = [
    "_build",
    "_generated",
    "_flight",
    "_exporter",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    # "custom.css", # Add custom CSS later if needed
    "right-sidebar.css",
]
html_js_files = [
    "toc-cleanup.js",  # Remove class prefixes from right sidebar TOC
]
html_show_sphinx = False

# Syntax highlighting
pygments_style = "default"  # Light theme
pygments_dark_style = "monokai"  # Dark theme - better for code readability

# PyData theme options
html_theme_options = {
    "logo": {
        "text": "Arize AX Client",
        "image_light": "logo.png",
        "image_dark": "logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Arize-ai/client_python",
            "icon": "fa-brands fa-github",
        },
    ],
    "external_links": [
        {"name": "Main Docs", "url": "https://docs.arize.com"},
        {"name": "Status Page", "url": "https://status.arize.com"},
    ],
    "navbar_align": "content",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["custom-tabs.html"],  # Custom tabs: Client | Observability | Data | Embeddings | Types
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "secondary_sidebar_items": ["page-toc"],  # Show page TOC in right sidebar
    "show_nav_level": 2,  # Show navigation in sidebar
    "footer_start": [],
    "footer_end": ["copyright"],
}

# Explicitly set sidebar templates
html_sidebars = {
    "**": ["custom-left-navbar.html"],  # Use custom template with full navigation
    "index": [],  # No left sidebar on index page
}

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}
