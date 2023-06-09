"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup ----------------------------------------------------------------
from datetime import datetime

# -- Project information -------------------------------------------------------
project = "sampling-handler"
author = "Andreas Vollrath"
copyright = f"2020-{datetime.now().year}, {author}"
release = "0.0.0"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinxcontrib.icon",
    "sphinxcontrib.btn",
]
exclude_patterns = ["**.ipynb_checkpoints"]
templates_path = ["_template"]

# -- Options for HTML output ---------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "text": project,
        "image_light": "_static/logo_light.png",
        "image_dark": "static_/logo_dark.png",
    },
    "use_edit_page_button": True,
    "footer_end": ["theme-version", "pypackage-credit"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/BuddyVolly/sampling-handler",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Pypi",
            "url": "https://pypi.org/project/sampling-handler/",
            "icon": "fa-brands fa-python",
        },
    ],
}
html_context = {
    "github_user": "BuddyVolly",
    "github_repo": "sampling-handler",
    "github_version": "main",
    "doc_path": "docs",
}
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
autoclass_content = "init"
autodoc_typehints = "description"
