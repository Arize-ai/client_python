---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for arize,
      with links to the rest of the site..
html_theme.sidebar_secondary.remove: true
---

# Arize Python SDK Reference

<br/>
<div id="external-links">
  <a href="https://docs.arize.com/arize/">
      <img src="https://img.shields.io/static/v1?message=Docs&logo=data:image/png;base64,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&labelColor=grey&color=blue&logoColor=white&label=%20"/>
  </a>
  <a target="_blank" class="link" href="https://join.slack.com/t/arize-ai/shared_invite/zt-1px8dcmlf-fmThhDFD_V_48oU7ALan4Q">
      <img src="https://img.shields.io/static/v1?message=Community&logo=slack&labelColor=grey&color=blue&logoColor=white&label=%20"/>
  </a>
  <a target="_blank" class="link" href="https://pypi.org/project/arize/">
      <img src="https://img.shields.io/pypi/v/arize?color=blue">
  </a>
  <a target="_blank" class="link" href="https://pypi.org/project/arize/">
      <img src="https://img.shields.io/pypi/pyversions/arize">
  </a>
</div>
<br/>

Below you can find the classes, functions, and parameters for tracing and evaluating your application using the Arize Python SDK. To get a complete guide on how to use Arize, including tutorials, quickstarts, and concept explanations, read our [docs](https://docs.arize.com/arize). This primarily covers LLM SDK methods, and we're documenting the rest of the methods for the full SDK spanning ML and CV use cases soon!

```{seealso}
Check out our [Slack community](https://arize-ai.slack.com/join/shared_invite/zt-1px8dcmlf-fmThhDFD_V_48oU7ALan4Q#/shared-invite/email) and [GitHub repository](https://github.com/arize-ai/client_python)!
```

::::{grid} 2
:::{grid-item}
```{toctree}
:maxdepth: 1
:caption: LLM API
llm-api/datasets
llm-api/evaluators
llm-api/exporter
llm-api/logger
llm-api/types
```
:::
:::{grid-item}
```{toctree}
:maxdepth: 1
:caption: ML API
ml-api
ml-api/exporter
ml-api/logger
```
:::
::::




