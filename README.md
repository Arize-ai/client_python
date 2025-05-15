<p align="center">
  <a href="https://arize.com/ax">
    <img src="https://storage.googleapis.com/arize-assets/arize-logo-white.jpg" width="600" />
  </a>
  <a target="_blank" href="https://pypi.org/project/arize/">
    <img src="https://img.shields.io/pypi/v/arize?color=blue">
  </a>
  <a target="_blank" href="https://pypi.org/project/arize/">
      <img src="https://img.shields.io/pypi/pyversions/arize">
  </a>
  <a target="_blank" href="https://arize-ai.slack.com/join/shared_invite/zt-2w57bhem8-hq24MB6u7yE_ZF_ilOYSBw#/shared-invite/email">
    <img src="https://img.shields.io/badge/slack-@arize-blue.svg?logo=slack">
  </a>
</p>

---

## Overview

A helper package to interact with Arize AI APIs.

Arize is an AI engineering platform. It helps engineers develop, evaluate, and observe AI applications and agents. 

Arize has both Enterprise and OSS products to support this goal: 
- [Arize AX](https://arize.com/) — an enterprise AI engineering platform from development to production, with an embedded AI Copilot
- [Phoenix](https://github.com/Arize-ai/phoenix) — a lightweight, open-source project for tracing, prompt engineering, and evaluation
- [OpenInference](https://github.com/Arize-ai/openinference) — an open-source instrumentation package to trace LLM applications across models and frameworks

We log over 1 trillion inferences and spans, 10 million evaluation runs, and 2 million OSS downloads every month. 

## Key Features
- [**_Tracing_**](https://docs.arize.com/arize/observe/tracing) - Trace your LLM application's runtime using OpenTelemetry-based instrumentation.
- [**_Evaluation_**](https://docs.arize.com/arize/evaluate/online-evals) - Leverage LLMs to benchmark your application's performance using response and retrieval evals.
- [**_Datasets_**](https://docs.arize.com/arize/develop/datasets) - Create versioned datasets of examples for experimentation, evaluation, and fine-tuning.
- [**_Experiments_**](https://docs.arize.com/arize/develop/datasets-and-experiments) - Track and evaluate changes to prompts, LLMs, and retrieval.
- [**_Playground_**](https://docs.arize.com/arize/develop/prompt-playground)- Optimize prompts, compare models, adjust parameters, and replay traced LLM calls.
- [**_Prompt Management_**](https://docs.arize.com/arize/develop/prompt-hub)- Manage and test prompt changes systematically using version control, tagging, and experimentation.

## Installation

Install Arize via `pip` or `conda`:

```bash
pip install arize
```

Install the `arize-otel` package for auto-instrumentation of your LLM library:

```bash
pip install arize-otel
```

## Usage
  
### Instrumentation
See https://pypi.org/project/arize-otel/

```python
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

# Setup OpenTelemetry via our convenience function
tracer_provider = register(
    space_id=SPACE_ID,
    api_key=API_KEY,
    project_name="agents-cookbook",
)

# Start instrumentation
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```


### Logging Spans, Evaluations, and Annotations

Use `arize.pandas.logger` to log spans, evaluations, and annotations in bulk. See https://arize-client-python.readthedocs.io/en/latest/llm-api/logger.html

```python
from arize.pandas.logger import Client

arize_client = Client(
    space_key=os.environ["ARIZE_SPACE_KEY"],
    api_key=os.environ["ARIZE_API_KEY"],
)

arize_client.log_spans(
    dataframe=spans_df,
    project_name="your-llm-project",
)

arize_client.log_evaluations_sync(
    dataframe=evals_df,
    project_name="your-llm-project",
)

arize_client.log_annotations(
    dataframe=annotations_df,
    project_name="your-llm-project",
)
```

### Datasets & Experiments

Use `arize.experimental.datasets` to create datasets and run experiments. See https://arize-client-python.readthedocs.io/en/latest/llm-api/datasets.html

```python
from arize.experimental.datasets import ArizeDatasetsClient

datasets_client = ArizeDatasetsClient(api_key=os.environ["ARIZE_API_KEY"])

dataset_id = datasets_client.create_dataset(
    space_id=os.environ["ARIZE_SPACE_KEY"],
    dataset_name="llm-span-dataset",
    data=spans_df,
)
```


## Tracing Integrations

Arize is built on top of OpenTelemetry and is vendor, language, and framework agnostic. For details about tracing integrations and example applications, see the [OpenInference](https://github.com/Arize-ai/openinference) project.

**Python Integrations**
| Integration | Package | Version Badge |
|------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [OpenAI](https://docs.arize.com/phoenix/tracing/integrations-tracing/openai) | `openinference-instrumentation-openai` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-openai.svg)](https://pypi.python.org/pypi/openinference-instrumentation-openai) |
| [OpenAI Agents](https://docs.arize.com/phoenix/tracing/integrations-tracing/openai-agents-sdk) | `openinference-instrumentation-openai-agents` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-openai-agents.svg)](https://pypi.python.org/pypi/openinference-instrumentation-openai-agents) |
| [LlamaIndex](https://docs.arize.com/phoenix/tracing/integrations-tracing/llamaindex) | `openinference-instrumentation-llama-index` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-llama-index.svg)](https://pypi.python.org/pypi/openinference-instrumentation-llama-index) |
| [DSPy](https://docs.arize.com/phoenix/tracing/integrations-tracing/dspy) | `openinference-instrumentation-dspy` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-dspy.svg)](https://pypi.python.org/pypi/openinference-instrumentation-dspy) |
| [AWS Bedrock](https://docs.arize.com/phoenix/tracing/integrations-tracing/bedrock) | `openinference-instrumentation-bedrock` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-bedrock.svg)](https://pypi.python.org/pypi/openinference-instrumentation-bedrock) |
| [LangChain](https://docs.arize.com/phoenix/tracing/integrations-tracing/langchain) | `openinference-instrumentation-langchain` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-langchain.svg)](https://pypi.python.org/pypi/openinference-instrumentation-langchain) |
| [MistralAI](https://docs.arize.com/phoenix/tracing/integrations-tracing/mistralai) | `openinference-instrumentation-mistralai` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-mistralai.svg)](https://pypi.python.org/pypi/openinference-instrumentation-mistralai) |
| [Google GenAI](https://docs.arize.com/phoenix/tracing/integrations-tracing/google-gen-ai) | `openinference-instrumentation-google-genai` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-google-genai.svg)](https://pypi.python.org/pypi/openinference-instrumentation-google-genai) |
| [Guardrails](https://docs.arize.com/phoenix/tracing/integrations-tracing/guardrails) | `openinference-instrumentation-guardrails` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-guardrails.svg)](https://pypi.python.org/pypi/openinference-instrumentation-guardrails) |
| [VertexAI](https://docs.arize.com/phoenix/tracing/integrations-tracing/vertexai) | `openinference-instrumentation-vertexai` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-vertexai.svg)](https://pypi.python.org/pypi/openinference-instrumentation-vertexai) |
| [CrewAI](https://docs.arize.com/phoenix/tracing/integrations-tracing/crewai) | `openinference-instrumentation-crewai` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-crewai.svg)](https://pypi.python.org/pypi/openinference-instrumentation-crewai) |
| [Haystack](https://docs.arize.com/phoenix/tracing/integrations-tracing/haystack) | `openinference-instrumentation-haystack` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-haystack.svg)](https://pypi.python.org/pypi/openinference-instrumentation-haystack) |
| [LiteLLM](https://docs.arize.com/phoenix/tracing/integrations-tracing/litellm) | `openinference-instrumentation-litellm` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-litellm.svg)](https://pypi.python.org/pypi/openinference-instrumentation-litellm) |
| [Groq](https://docs.arize.com/phoenix/tracing/integrations-tracing/groq) | `openinference-instrumentation-groq` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-groq.svg)](https://pypi.python.org/pypi/openinference-instrumentation-groq) |
| [Instructor](https://docs.arize.com/phoenix/tracing/integrations-tracing/instructor) | `openinference-instrumentation-instructor` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-instructor.svg)](https://pypi.python.org/pypi/openinference-instrumentation-instructor) |
| [Anthropic](https://docs.arize.com/phoenix/tracing/integrations-tracing/anthropic) | `openinference-instrumentation-anthropic` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-anthropic.svg)](https://pypi.python.org/pypi/openinference-instrumentation-anthropic) |
| [Smolagents](https://huggingface.co/docs/smolagents/en/tutorials/inspect_runs) | `openinference-instrumentation-smolagents` | [![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-smolagents.svg)](https://pypi.python.org/pypi/openinference-instrumentation-smolagents) |

### JavaScript Integrations

| Integration                                                                                | Package                                            | Version Badge                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [OpenAI](https://docs.arize.com/phoenix/tracing/integrations-tracing/openai-node-sdk)      | `@arizeai/openinference-instrumentation-openai`    | [![NPM Version](https://img.shields.io/npm/v/@arizeai/openinference-instrumentation-openai.svg)](https://www.npmjs.com/package/@arizeai/openinference-instrumentation-openai)       |
| [LangChain.js](https://docs.arize.com/phoenix/tracing/integrations-tracing/langchain.js)   | `@arizeai/openinference-instrumentation-langchain` | [![NPM Version](https://img.shields.io/npm/v/@arizeai/openinference-instrumentation-langchain.svg)](https://www.npmjs.com/package/@arizeai/openinference-instrumentation-langchain) |
| [Vercel AI SDK](https://docs.arize.com/phoenix/tracing/integrations-tracing/vercel-ai-sdk) | `@arizeai/openinference-vercel`                    | [![NPM Version](https://img.shields.io/npm/v/@arizeai/openinference-vercel)](https://www.npmjs.com/package/@arizeai/openinference-vercel)                                           |
| [BeeAI](https://docs.arize.com/phoenix/tracing/integrations-tracing/beeai)                 | `@arizeai/openinference-instrumentation-beeai`     | [![NPM Version](https://img.shields.io/npm/v/@arizeai/openinference-vercel)](https://www.npmjs.com/package/@arizeai/openinference-instrumentation-beeai)                            |


## Community

Join our community to connect with thousands of AI builders.

- 🌍 Join our [Slack community](https://arize-ai.slack.com/join/shared_invite/zt-11t1vbu4x-xkBIHmOREQnYnYDH1GDfCg?__hstc=259489365.a667dfafcfa0169c8aee4178d115dc81.1733501603539.1733501603539.1733501603539.1&__hssc=259489365.1.1733501603539&__hsfp=3822854628&submissionGuid=381a0676-8f38-437b-96f2-fc10875658df#/shared-invite/email).
- 📚 Read our [documentation](https://docs.arize.com/arize).
- 💡 Ask questions and provide feedback in the _#arize-support_ channel.
- 𝕏 Follow us on [𝕏](https://twitter.com/ArizeAI).
- 🧑‍🏫 Deep dive into everything [Agents](http://arize.com/ai-agents/) and [LLM Evaluations](https://arize.com/llm-evaluation) on Arize's Learning Hubs.

Copyright 2025 Arize AI, Inc. All Rights Reserved.
