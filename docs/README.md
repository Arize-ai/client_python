# Arize Python SDK API Reference

This directory contains the Sphinx-based API reference documentation for the Arize Python SDK. This reference provides comprehensive details for the SDK's public API.

## Structure

```
docs/
 ┣ source/
 ┃ ┣ _static/
 ┃ ┃ └── custom.css
 ┃ ┣ client.md          # ArizeClient
 ┃ ┣ spans.md           # SpansClient
 ┃ ┣ ml.md              # MLModelsClient
 ┃ ┣ datasets.md        # DatasetsClient
 ┃ ┣ experiments.md     # ExperimentsClient
 ┃ ┣ embeddings.md      # EmbeddingGenerator
 ┃ ┣ types.md           # Type definitions
 ┃ ┣ config.md          # SDKConfiguration
 ┃ ┣ conf.py            # Sphinx configuration
 ┃ └── index.md         # Main entry point
 ┣ Makefile
 ┣ make.bat
 ┣ requirements.txt
 └── README.md

```

- **conf.py**: Sphinx configuration for theme, extensions, and autodoc settings
- **index.md**: Main entry point with table of contents
- **requirements.txt**: Python dependencies for building the documentation
- **Makefiles**: Commands for building HTML documentation locally

## Building the Documentation

### Prerequisites

Install dependencies:

```bash
pip install -e '.[docs,otel,mimic,embeddings]'
```

### Build HTML

```bash
cd docs
make clean
make html
```

The generated HTML will be in `build/html/`. Open `build/html/index.html` in your browser.

### Live Reload

For development with auto-reload:

```bash
make livehtml
```

## Configuration

### Theme

The documentation uses the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/), which provides:

- Clean, modern design
- Responsive layout
- Good navigation

### Autodoc

Documentation is automatically generated from Google-style docstrings in the source code using Sphinx's autodoc extension.

Configuration settings:

- `autodoc_typehints = "none"` - Types documented in parameter descriptions
- `autoclass_content = "class"` - Only class docstring, not **init**
- `add_module_names = False` - Cleaner class names

### MyST Parser

Markdown files are processed using [MyST Parser](https://myst-parser.readthedocs.io/), allowing:

- Markdown syntax for documentation
- Sphinx directives with `{eval-rst}` blocks
- Cross-references and linking

## Adding New API Documentation

1. Create a new `.md` file in `source/` directory
2. Add autodoc directives:

   ````markdown
   # MyClass

   ```{eval-rst}
   .. currentmodule:: arize.mymodule

   .. autoclass:: MyClass
      :members:
      :undoc-members:
      :show-inheritance:
      :special-members: __init__
   ```
   ````

   ```

   ```

3. Add the file to the toctree in `index.md`:

   ````markdown
   ```{toctree}
   :maxdepth: 2

   client
   myclass  # <- Add here
   ```
   ````

   ```

   ```

4. Rebuild the documentation

## Publishing

For ReadTheDocs integration, create a `.readthedocs.yaml` file in the repository root (see plan document for template).
