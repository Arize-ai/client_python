<p align="center">
  <a href="https://arize.com/ax">
    <img src="https://storage.googleapis.com/arize-assets/arize-logo-white.jpg" width="600" />
  </a>
  <br/>
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

# Table of Contents <!-- omit in toc -->

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Optional Dependencies](#optional-dependencies)
  - [Migrating from Version 7](#migrating-from-version-7)
- [Usage](#usage)
  - [Instrumentation](#instrumentation)
  - [Operations on Spans](#operations-on-spans)
    - [Logging spans](#logging-spans)
    - [Update spans Evaluations, Annotations, and Metadata](#update-spans-evaluations-annotations-and-metadata)
    - [Exporting spans](#exporting-spans)
  - [Operations on ML Models](#operations-on-ml-models)
    - [Stream log ML Data for a Classification use-case](#stream-log-ml-data-for-a-classification-use-case)
    - [Log a batch of ML Data for a Object Detection use-case](#log-a-batch-of-ml-data-for-a-object-detection-use-case)
    - [Exporting ML Data](#exporting-ml-data)
  - [Generate embeddings for your data](#generate-embeddings-for-your-data)
  - [Operations on Datasets](#operations-on-datasets)
    - [List Datasets](#list-datasets)
    - [Create a Dataset](#create-a-dataset)
    - [Get Dataset](#get-dataset)
    - [Delete a Dataset](#delete-a-dataset)
    - [List Dataset Examples](#list-dataset-examples)
  - [Operations on Experiments](#operations-on-experiments)
    - [List Experiments](#list-experiments)
    - [Run an Experiment](#run-an-experiment)
    - [Create an Experiment](#create-an-experiment)
    - [Get an Experiment](#get-an-experiment)
    - [Delete an Experiment](#delete-an-experiment)
    - [List Experiment runs](#list-experiment-runs)
- [SDK Configuration](#sdk-configuration)
  - [Logging](#logging)
    - [In Code](#in-code)
    - [Via Environment Variables](#via-environment-variables)
  - [Caching](#caching)
    - [In Code](#in-code-1)
    - [Via Environment Variables](#via-environment-variables-1)
    - [Clean the cache](#clean-the-cache)
- [Community](#community)

# Overview

A helper package to interact with Arize AI APIs.

Arize is an AI engineering platform. It helps engineers develop, evaluate, and observe AI applications and agents.

Arize has both Enterprise and OSS products to support this goal:

- [Arize AX](https://arize.com/) — an enterprise AI engineering platform from development to production, with an embedded AI Copilot
- [Phoenix](https://github.com/Arize-ai/phoenix) — a lightweight, open-source project for tracing, prompt engineering, and evaluation
- [OpenInference](https://github.com/Arize-ai/openinference) — an open-source instrumentation package to trace LLM applications across models and frameworks

We log over 1 trillion inferences and spans, 10 million evaluation runs, and 2 million OSS downloads every month.

# Key Features

- [**_Tracing_**](https://docs.arize.com/arize/observe/tracing) - Trace your LLM application's runtime using OpenTelemetry-based instrumentation.
- [**_Evaluation_**](https://docs.arize.com/arize/evaluate/online-evals) - Leverage LLMs to benchmark your application's performance using response and retrieval evals.
- [**_Datasets_**](https://docs.arize.com/arize/develop/datasets) - Create versioned datasets of examples for experimentation, evaluation, and fine-tuning.
- [**_Experiments_**](https://docs.arize.com/arize/develop/datasets-and-experiments) - Track and evaluate changes to prompts, LLMs, and retrieval.
- [**_Playground_**](https://docs.arize.com/arize/develop/prompt-playground)- Optimize prompts, compare models, adjust parameters, and replay traced LLM calls.
- [**_Prompt Management_**](https://docs.arize.com/arize/develop/prompt-hub)- Manage and test prompt changes systematically using version control, tagging, and experimentation.

# Installation

Install the base package:

```bash
pip install arize
```

## Optional Dependencies

The following optional extras provide specialized functionality:

> **Note:** The `otel` extra installs the `arize-otel` package, which is also available as a standalone package. If you only need auto-instrumentation without the full SDK, install `arize-otel` directly.

| Extra | Install Command | What It Provides |
|-------|----------------|------------------|
| **otel** | `pip install arize[otel]` | OpenTelemetry auto-instrumentation package (arize-otel) for automatic tracing |
| **embeddings** | `pip install arize[embeddings]` | Automatic embedding generation for NLP, CV, and structured data (Pillow, datasets, tokenizers, torch, transformers) |
| **mimic** | `pip install arize[mimic]` | MIMIC explainer for model interpretability |

Install multiple extras:

```bash
pip install arize[otel,embeddings,mimic]
```

## Migrating from Version 7

If you're upgrading from version 7, please refer to the [Migration Guide](https://arize.com/docs/api-clients/python/version-8/migration) for detailed migration steps and breaking changes.

# Usage

## Instrumentation

See [arize-otel in PyPI](https://pypi.org/project/arize-otel/):

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

## Operations on Spans

Use `arize.spans` to interact with spans: log spans into Arize, update the span's
evaluations, annotations and metadata in bulk.

### Logging spans

```python
from arize import ArizeClient

client = ArizeClient(api_key=API_KEY)
SPACE_ID = "<your-space-id>"
PROJECT_NAME = "<your-project-name>"

client.spans.log(
    space_id=SPACE_ID,
    project_name=PROJECT_NAME,
    dataframe=spans_df,
    # evals_df=evals_df, # Optionally pass the evaluations together with the spans
)
```

### Update spans Evaluations, Annotations, and Metadata

```python
from arize import ArizeClient

client = ArizeClient(api_key=API_KEY)
SPACE_ID = "<your-space-id>"
PROJECT_NAME = "<your-project-name>"

client.spans.update_evaluations(
    space_id=SPACE_ID,
    project_name=PROJECT_NAME,
    dataframe=evals_df,
    # force_http=... # Optionally pass force_http to update evaluations via HTTP instead of gRPC, defaults to False
)

client.spans.update_annotations(
    space_id=SPACE_ID,
    project_name=PROJECT_NAME,
    dataframe=annotations_df,
)

client.spans.update_metadata(
    space_id=SPACE_ID,
    project_name=PROJECT_NAME,
    dataframe=metadata_df,
)
```

### Exporting spans

Use the `export_to_df` or `export_to_parquet` to export large amounts of spans from Arize.

```python
from arize import ArizeClient
from datetime import datetime

FMT  = "%Y-%m-%d"
start_time = datetime.strptime("2024-01-01",FMT)
end_time = datetime.strptime("2026-01-01",FMT)

client = ArizeClient(api_key=API_KEY)
SPACE_ID = "<your-space-id>"
PROJECT_NAME = "<your-project-name>"

df = client.spans.export_to_df(
    space_id=SPACE_ID,
    project_name=PROJECT_NAME,
    start_time=start_time,
    end_time=end_time,
)
```

## Operations on ML Models

Use `arize.ml` to interact with ML models: log ML data (training, validation, production)
into Arize, either streaming or in batches.

### Stream log ML Data for a Classification use-case

```python
from arize import ArizeClient
from arize.ml.types import ModelTypes, Environments

client = ArizeClient(api_key=API_KEY)
SPACE_ID = "<your-space-id>"
MODEL_NAME = "<your-model-name>"

features=...
embedding_features=...

response = client.ml.log_stream(
    space_id=SPACE_ID,
    model_name=MODEL_NAME,
    model_type=ModelTypes.SCORE_CATEGORICAL,
    environment=Environments.PRODUCTION,
    prediction_label=("not fraud",0.3),
    actual_label=("fraud",1.0),
    features=features,
    embedding_features=embedding_features,
)
```

### Log a batch of ML Data for a Object Detection use-case

```python
from arize import ArizeClient
from arize.ml.types import ModelTypes, Environments

client = ArizeClient(api_key=API_KEY)
SPACE_ID = "<your-space-id>"
MODEL_NAME = "<your-model-name>"
MODEL_VERSION = "1.0"

from arize.ml.types import Schema, EmbeddingColumnNames, ObjectDetectionColumnNames, ModelTypes, Environments

tags = ["drift_type"]
embedding_feature_column_names = {
    "image_embedding": EmbeddingColumnNames(
        vector_column_name="image_vector", link_to_data_column_name="url"
    )
}
object_detection_prediction_column_names = ObjectDetectionColumnNames(
    bounding_boxes_coordinates_column_name="prediction_bboxes",
    categories_column_name="prediction_categories",
    scores_column_name="prediction_scores",
)
object_detection_actual_column_names = ObjectDetectionColumnNames(
    bounding_boxes_coordinates_column_name="actual_bboxes",
    categories_column_name="actual_categories",
)

# Define a Schema() object for Arize to pick up data from the correct columns for logging
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="prediction_ts",
    tag_column_names=tags,
    embedding_feature_column_names=embedding_feature_column_names,
    object_detection_prediction_column_names=object_detection_prediction_column_names,
    object_detection_actual_column_names=object_detection_actual_column_names,
)

# Logging Production DataFrame
response = client.ml.log_batch(
    space_id=SPACE_ID,
    model_name=MODEL_NAME,
    model_type=ModelTypes.OBJECT_DETECTION,
    dataframe=prod_df,
    schema=schema,
    environment=Environments.PRODUCTION,
    model_version = MODEL_VERSION, # Optionally pass a model version
)
```

### Exporting ML Data

Use the `export_to_df` or `export_to_parquet` to export large amounts of spans from Arize.

```python
from arize import ArizeClient
from datetime import datetime

FMT  = "%Y-%m-%d"
start_time = datetime.strptime("2024-01-01",FMT)
end_time = datetime.strptime("2026-01-01",FMT)

client = ArizeClient(api_key=API_KEY)
SPACE_ID = "<your-space-id>"
MODEL_NAME = "<your-model-name>"
MODEL_VERSION = "1.0"

df = client.ml.export_to_df(
    space_id=SPACE_ID,
    model_name=MODEL_NAME,
    environment=Environments.TRAINING,
    model_version=MODEL_VERSION,
    start_time=start_time,
    end_time=end_time,
)
```

## Generate embeddings for your data

```python
import pandas as pd
from arize.embeddings import EmbeddingGenerator, UseCases

# You can check available models
print(EmbeddingGenerator.list_pretrained_models())

# Example dataframe
df = pd.DataFrame(
    {
        "text": [
            "Hello world.",
            "Artificial Intelligence is the future.",
            "Spain won the FIFA World Cup on 2010.",
        ],
    }
)
# Instantiate the generator for your usecase, selecting the base model
generator = EmbeddingGenerator.from_use_case(
    use_case=UseCases.NLP.SEQUENCE_CLASSIFICATION,
    model_name="distilbert-base-uncased",
    tokenizer_max_length=512,
    batch_size=100,
)

# Generate embeddings
df["text_vector"] = generator.generate_embeddings(text_col=df["text"])
```

## Operations on Datasets

### List Datasets

You can list all datasets that the user has access to using `client.datasets.list()`. You can use the `limit` parameter to specify the maximum number of datasets desired in the response and you can specify the `space_id` to target the list operation to a particular space.

```python
resp = client.datasets.list(
    limit=... # Optional
    space_id=... # Optional
)
```

The response is an object of type `DatasetsList200Response`, and you can access the list of datasets via its `datasets` attribute. In addition, you can transform the response object to a dictionary, to JSON format, or a pandas dataframe.

```python
# Get the list of datasets from the response
dataset_list = resp.datasets
# Get the response as a dictionary
resp_dict = resp.to_dict()
# Get the response in JSON format
resp_json = resp.to_json()
# Get the response as a pandas dataframe
resp_df = resp.to_df()
```

### Create a Dataset

You can create a dataset using `client.datasets.create()`. You must pass examples, we currently don't support creating an empty dataset, for instance, these are 2 rows of examples, as a list of dictionaries. You can also pass a pandas dataframe for the examples.

```python
examples = [
    {
        "eval.Correctness Basic.explanation": "The query indicates that the user is having trouble accessing their account on their laptop, while access on their phone is still working. This suggests a potential issue with the login process on the laptop, which aligns with the 'Login Issues' queue. The mention of a possible change in the account could relate to login credentials or settings affecting the laptop specifically, but it still falls under the broader category of login issues.",
        "eval.Correctness Basic.label": "correct",
        "eval.Correctness Basic.score": 1,
        "llm output": "Login Issues",
        "query": "I can't get in on my laptop anymore, but my phone still works fine — could this be because I changed something in my account?"
    },
    {
        "eval.Correctness Basic.explanation": "The query is about a user who signed up but is unable to log in because the system says no account is found. This issue is related to the login process, as the user is trying to access their account and is facing a problem with the login system recognizing their account. Therefore, assigning this query to the 'Login Issues' queue is appropriate.",
        "eval.Correctness Basic.label": "correct",
        "eval.Correctness Basic.score": 1,
        "llm output": "Login Issues",
        "query": "Signed up ages ago but never got around to logging in — now it says no account found. Do I start over?"
    }
]
```

If the number of examples (rows in dataframe, items in list) is too large, the client SDK will try to send the data via Arrow Flight via gRPC for better performance. If you want to force the data transfer to HTTP you can use the `force_http` flag. The response is a `Dataset` object.

```python
created_dataset = client.datasets.create(
    space_id="<target-space-id>",
    name="<your-dataset-name>", # Name must be unique within a space
    examples=..., # List of dictionaries or pandas dataframe
    # force_http=... # Optionally pass force_http to create datasets via HTTP instead of gRPC, defaults to False
)
```

The `Dataset` object also counts with convenience method similar to `List***` objects:

```python
# Get the response as a dictionary
dataset_dict = create_dataset.to_dict()
# Get the response in JSON format
dataset_dict = create_dataset.to_json()
```

### Get Dataset

To get a dataset by its ID use `client.datasets.get()`, you can optionally also pass the version ID of a particular version of interest of the dataset. The returned type is `Dataset`.

```python
dataset = client.datasets.get(
    dataset_id=... # The unique identifier of the dataset
    dataset_version_id=... # The unique identifier of the dataset version
)
```

### Delete a Dataset

To delete a dataset by its ID use `client.datasets.delete()`. The call returns `None` if successful deletion took place, error otherwise.

```python
client.datasets.delete(
    dataset_id=... # The unique identifier of the dataset
)
```

### List Dataset Examples

You can list the examples of a given dataset using `client.datasets.list_examples()` and passing the dataset ID and, optionally, the dataset version ID. You can specify the number of examples desired using the `limit` parameter. If you want a large number of examples, consider using the `all=True` parameter, which will make it so the SDK exports the data using Arrow Flight via gRPC, for increased performance.

```python
resp = client.datasets.list_examples(
    dataset_id="your-dataset-id>",
    dataset_version_id="your-dataset-version-id>", # Optional, defaults to latest version
    limit=... # number of desired examples. Defaults to 100
    all=... # Whether or not to export all of the examples. Defaults to False
)
```

The response is an object of type `DatasetsExamplesList200Response`, and you can access the list of examples via its `examples` attribute. In addition, you can transform the response object to a dictionary, to JSON format, or a pandas dataframe.

```python
# Get the list of datasets from the response
examples_list = resp.examples
# Get the response as a dictionary
resp_dict = resp.to_dict()
# Get the response in JSON format
resp_json = resp.to_json()
# Get the response as a pandas dataframe
resp_df = resp.to_df()
```

## Operations on Experiments

### List Experiments

You can list all experiments that the user has access to using `client.experiments.list()`. You can use the `limit` parameter to specify the maximum number of datasets desired in the response and you can specify the `dataset_id` to target the list operation to a particular dataset.

```python
resp = client.experiments.list(
    limit=... # Optional
    dataset_id=... # Optional
)
```

The response is an object of type `ExperimentsList200Response`, and you can access the list of experiments via its `experiments` attribute. In addition, you can transform the response object to a dictionary, to JSON format, or a pandas dataframe.

```python
# Get the list of datasets from the response
experiment_list = resp.experiments
# Get the response as a dictionary
resp_dict = resp.to_dict()
# Get the response in JSON format
resp_json = resp.to_json()
# Get the response as a pandas dataframe
resp_df = resp.to_df()
```

### Run an Experiment

You can run an experiment on a dataset using `client.experiments.run()` by defining a task, evaluators (optional), and passing the dataset id of the dataset you want to use, together with a name for the experiment. The function will download the entire dataset from Arize (unless cached, see caching section under "SDK Configuration"), execute the task to obtain an output, and perform evaluations (if evaluators were passed). The experiments will also be traced, and these traces will be visible in Arize. The experiment will be created and the data logged into Arize automatically. You can avoid logging to Arize by making `dry_run=True`. The function will return the `Experiment` object (or `None` if `dry_run=True`) together with the dataframe with the experiment data.

```python
experiment, experiment_df = client.run_experiment(
    name="<name-your-experiment>",
    dataset_id="<id-of-dataset-to-use>",
    task=... # The task to be performed in the experiment.
    evaluators=... # Optional: The evaluators to use in the experiment.
    dry_run=..., # If True, the experiment result will not be uploaded to Arize. Defaults to False
    dry_run_count=..., # Number of examples of the dataset to use in the dry run. Defaults to 10
    concurrency=..., # The number of concurrent tasks to run. Defaults to 3.
    set_global_tracer_provider=..., # If True, sets the global tracer provider for the experiment. Defaults to False
    exit_on_error=..., # If True, the experiment will stop running on first occurrence of an error. Defaults to False
)
```

The `Experiment` object also counts with convenience method similar to `List***` objects:

```python
# Get the response as a dictionary
experiment_dict = create_experiment.to_dict()
# Get the response in JSON format
experiment_dict = create_experiment.to_json()
```

### Create an Experiment

It is possible that you have run the experiment yourself without the above function, and hence you already have experiment data that you want to send to Arize. In this case, use the `client.experiments.create()` method by passing the runs data, we currently don't support creating an empty experiment, for instance, these are 2 rows of runs, as a list of dictionaries. You can also pass a pandas dataframe for the runs data.

> NOTE: If you don't have experiment data and want to run experiment, see the `client.experiments.run()` section above.

```python
# TODO
runs = [
]
```

In addition, you must specify which columns are the `example_id` and the `result`, you can do so by using the `ExperimentTaskResultFieldNames`. Moreover, if you choose to pass evaluation data, you can indicate the evaluation columns using `EvaluationResultFieldNames`:

```python
# TODO
```

If the number of runs (rows in dataframe, items in list) is too large, the client SDK will try to send the data via Arrow Flight via gRPC for better performance. If you want to force the data transfer to HTTP you can use the `force_http` flag. The response is an `Experiment` object.

```python
created_experiment = client.experiments.create(
    name="<your-experiment-name>", # Name must be unique within a dataset
    dataset_id="<desired-dataset-id>",
    experiment_runs=..., # List of dictionaries or pandas dataframe
    task_fields=ExperimentTaskResultFieldNames(...),
    evaluator_columns=... # Optional
    # force_http=... # Optionally pass force_http to create experiments via HTTP instead of gRPC, defaults to False
)
```

### Get an Experiment

To get a dataset by its ID use `client.datasets.get()`, you can optionally also pass the version ID of a particular version of interest of the dataset. The returned type is `Dataset`.

```python
dataset = client.datasets.get(
    dataset_id=... # The unique identifier of the dataset
    dataset_version_id=... # The unique identifier of the dataset version
)
```

### Delete an Experiment

To delete an experiment by its ID use `client.experiments.delete()`. The call returns `None` if successful deletion took place, error otherwise.

```python
client.experiments.delete(
    experiment_id=... # The unique identifier of the experiment
)
```

### List Experiment runs

You can list the runs of a given experiment using `client.experiments.list_runs()` and passing the experiment ID. You can specify the number of runs desired using the `limit` parameter. If you want a large number of runs, consider using the `all=True` parameter, which will make it so the SDK exports the data using Arrow Flight via gRPC, for increased performance.

```python
resp = client.experiments.list_runs(
    experiment_id="your-experiment-id>",
    limit=... # number of desired runs. Defaults to 100
    all=... # Whether or not to export all of the runs. Defaults to False
)
```

The response is an object of type `ExperimentsRunsList200Response`, and you can access the list of runs via its `experiment_runs` attribute. In addition, you can transform the response object to a dictionary, to JSON format, or a pandas dataframe.

```python
# Get the list of datasets from the response
run_list = resp.experiments_runs
# Get the response as a dictionary
resp_dict = resp.to_dict()
# Get the response in JSON format
resp_json = resp.to_json()
# Get the response as a pandas dataframe
resp_df = resp.to_df()
```

# SDK Configuration

## Logging

### In Code

You can use `configure_logging` to set up the logging behavior of the Arize package to your needs.

```python
from arize.logging import configure_logging

configure_logging(
    level=..., # Defaults to logging.INFO
    structured=..., # if True, emit JSON logs. Defaults to False
)
```

### Via Environment Variables

Configure the same options as the section above, via:

```python
import os

# Whether or not you want to disable logging altogether
os.environ["ARIZE_LOG_ENABLE"] = "true"
# Set up the logging level
os.environ["ARIZE_LOG_LEVEL"] = "debug"
# Whether or not you want structured JSON logs
os.environ["ARIZE_LOG_STRUCTURED"] = "false"
```

The default behavior of Arize's logs is: enabled, `INFO` level, and not structured.

## Caching

When downloading big segments of data from Arize, such as a `Dataset` with all of its examples, the SDK will cache the file in `parquet` format under `~/.arize/cache/datasets/dataset_<updated_at_timestamp>.parquet`.

### In Code

You can disable caching via the `enable_caching` parameter when instantiating the client, and also edit the "arize directory":

```python
client = ArizeClient(
    enable_caching=False, # Optional parameter, defaults to True
    arize_directory="my-desired-directory", # Optional parameter, defaults to ~/.arize
)
```

### Via Environment Variables

You can also configure the above via:

```python
import os

# Whether or not you want to disable caching
os.environ["ARIZE_ENABLE_CACHING"] = "true"
# Where you want the SDK to store the files
os.environ["ARIZE_DIRECTORY"] = "~/.arize"
```

### Clean the cache

To clean the cache you can directly `rm` the files or directory. In addition, the client has the option to help with that as well using `client.clear_cache()`, which will delete the `cache/` directory inside the _arize directory_ (defaults to `~/.arize`).

# Community

Join our community to connect with thousands of AI builders.

- 🌍 Join our [Slack community](https://arize-ai.slack.com/join/shared_invite/zt-11t1vbu4x-xkBIHmOREQnYnYDH1GDfCg?__hstc=259489365.a667dfafcfa0169c8aee4178d115dc81.1733501603539.1733501603539.1733501603539.1&__hssc=259489365.1.1733501603539&__hsfp=3822854628&submissionGuid=381a0676-8f38-437b-96f2-fc10875658df#/shared-invite/email).
- 📚 Read our [documentation](https://docs.arize.com/arize).
- 💡 Ask questions and provide feedback in the _#arize-support_ channel.
- 𝕏 Follow us on [𝕏](https://twitter.com/ArizeAI).
- 🧑‍🏫 Deep dive into everything [Agents](http://arize.com/ai-agents/) and [LLM Evaluations](https://arize.com/llm-evaluation) on Arize's Learning Hubs.

Copyright 2025 Arize AI, Inc. All Rights Reserved.
