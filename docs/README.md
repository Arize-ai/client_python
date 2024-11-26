# Maintenance README for Arize Sphinx API Documentation

This API reference provides comprehensive details for Arize's API. The documentation covers only public, user-facing API endpoints offered in Arize.

Maintaining the API reference consists of two parts:

1. Building the documentation with Sphinx
2. Hosting and CI with readthedocs

## TL;DR
```
uv venv --python=python3.11
uv pip install -r requirements.txt
make clean html
# then open build/html/index.html in your browser
# currently, the build/html directory is copied over as the static site for arize-docs.onrender.com
```

## Files
- conf.py: All sphinx-related configuration is done here and is necessary to run Sphinx.
- index.md: Main entrypoint for the API reference. This file must be in the `source` directory. For documentation to show up on the API reference, there must be a path (does not have to be direct) defined in index.md to the target documentation file.
- requirements.txt: This file is necessary for management of dependencies on the readthedocs platform and its build process.
- make files: Not required but useful in generating static HTML pages locally.

## Useful references
https://pydata-sphinx-theme.readthedocs.io/
https://sphinx-design.readthedocs.io/en/latest/
https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
https://docs.readthedocs.io/en/stable/automation-rules.html