# ArizeClient

The main entry point for the Arize SDK.

<!--
AUTODOC OPTIONS GUIDE
=====================

This file uses Sphinx autodoc directives to automatically generate documentation from docstrings.
Below are examples of all available options with explanations.

Global Defaults (set in conf.py):
- autodoc_default_options: Sets default options for all autodoc directives
- These can be overridden per-directive by specifying options explicitly

Common Patterns:
- Use :members:                    Include all public members (methods, attributes)
- Use :members: foo, bar           Include only specific members by name
- Use :undoc-members:              Include members without docstrings
- Use :private-members:            Include private members (starting with _)
- Use :special-members:            Include special members (__init__, __str__, etc.)
- Use :special-members: __init__   Include only specific special members
- Use :inherited-members:          Include members inherited from base classes
- Use :show-inheritance:           Show base classes this class inherits from
- Use :member-order: alphabetical  Sort members alphabetically (default)
- Use :member-order: bysource      Order members as they appear in source code
- Use :member-order: groupwise     Group by type first (attributes, methods, etc.), then alphabetically within each group
- Use :exclude-members: foo, bar   Exclude specific members from documentation
- Use :noindex:                    Don't add this to the index (use when documenting same thing twice)
- Use Type Hints Options:
- Use :no-typehints:               Don't show type hints

What to Document:
- Use :class-doc-from: class       Use only class docstring (not __init__)
- Use :class-doc-from: init        Use only __init__ docstring
- Use :class-doc-from: both        Concatenate class and __init__ docstrings
-->

```{eval-rst}
.. currentmodule:: arize
.. autoclass:: ArizeClient
   :members:
   :member-order: groupwise
```
