# Tasks

```{eval-rst}
.. currentmodule:: arize.tasks.client
.. autoclass:: TasksClient
   :members:
   :member-order: bysource
```

## Usage Examples

### Span-granularity task

Span tasks use a `query_filter` string at the task level and pass
`SpanEvaluatorInput` entries to `evaluators`.

```python
from arize.tasks import TasksClient, TaskType, SpanEvaluatorInput

client = TasksClient(...)

task = client.create_evaluation_task(
    name="Weekly Quality Check",
    task_type=TaskType.TEMPLATE_EVALUATION,
    space="my-space",
    project="my-project",
    query_filter="span_kind = 'LLM'",
    evaluators=[
        SpanEvaluatorInput(
            evaluator_id="your_evaluator_id",
            column_mappings={"input": "attributes.input.value", "output": "attributes.output.value"},
        )
    ],
)
```

### Trace/session-granularity task (multi-span query)

Trace/session tasks use `query_filters` (a `TaskQueryFilters` object with named
filters and an optional boolean expression) at the task level and pass
`TraceOrSessionEvaluatorInput` entries to `evaluators`.

```python
from arize.tasks import (
    TasksClient,
    TaskType,
    TaskQueryFilter,
    TaskQueryFilters,
    TaskQueryMapping,
    TraceOrSessionEvaluatorInput,
)

client = TasksClient(...)

task = client.create_evaluation_task(
    name="Trace Quality Check",
    task_type=TaskType.TEMPLATE_EVALUATION,
    space="my-space",
    project="my-project",
    query_filters=TaskQueryFilters(
        filters=[
            TaskQueryFilter(id="A", filter="span_kind = 'LLM'"),
            TaskQueryFilter(id="B", filter="span_kind = 'RETRIEVER'"),
        ],
        expression="A AND B",
    ),
    evaluators=[
        TraceOrSessionEvaluatorInput(
            evaluator_id="your_evaluator_id",
            query_mappings=[
                TaskQueryMapping(variable_name="input", query_ids=["A"], attribute_path="attributes.input.value"),
                TaskQueryMapping(variable_name="output", query_ids=["B"], attribute_path="attributes.output.value"),
            ],
        )
    ],
)
```

### Updating a task

Both `query_filter` and `query_filters` can be updated; they are mutually
exclusive — provide at most one.

```python
# Switch to trace/session shape
task = client.update(
    task="your_task_id",
    space="my-space",
    query_filters=TaskQueryFilters(
        filters=[TaskQueryFilter(id="A", filter="span_kind = 'LLM'")],
        expression="A",
    ),
    evaluators=[
        TraceOrSessionEvaluatorInput(
            evaluator_id="your_evaluator_id",
            query_mappings=[
                TaskQueryMapping(variable_name="input", query_ids=["A"], attribute_path="attributes.input.value"),
            ],
        )
    ],
)
```

## Response Types

```{eval-rst}
.. automodule:: arize.tasks.types
   :members:
   :member-order: bysource
```
