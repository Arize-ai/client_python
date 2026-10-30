# CreateTaskRequest

Request body for creating a task. The `type` field is the discriminator.  | `type` | Schema | |---|---| | `TEMPLATE_EVALUATION` | `CreateTemplateEvaluationTaskRequest` | | `CODE_EVALUATION` | `CreateCodeEvaluationTaskRequest` | | `RUN_EXPERIMENT` | `CreateRunExperimentTaskRequest` | | `DECISION_JUDGE_EVALUATION` | `CreateDecisionJudgeEvaluationTaskRequest` |  `RUN_EXPERIMENT` tasks do not run continuously — they must be triggered explicitly via `POST /v2/tasks/{task_id}/trigger` each time. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Task name | 
**project_id** | **str** | Project identifier (base64). Decision Judge tasks run on the spans of a project. Datasets are not supported.  | 
**dataset_id** | **str** | Dataset identifier (base64). Required for &#x60;RUN_EXPERIMENT&#x60; tasks. | 
**experiment_ids** | **List[str]** | Experiment identifiers (base64). Required when &#x60;dataset_id&#x60; is provided (at least one entry). Must be omitted or empty for project-based tasks.  | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously on incoming spans. | [optional] 
**query_filter** | **str** | Task-level query filter applied to all evaluated spans. | [optional] 
**query_filters** | [**TaskQueryFiltersInput**](TaskQueryFiltersInput.md) | Named query filters plus optional expression for trace/session evaluators. Mutually exclusive with &#x60;query_filter&#x60;.  | [optional] 
**evaluators** | [**List[SpanEvaluatorInput]**](SpanEvaluatorInput.md) | Decision Judge evaluators to attach (at least one required). Every evaluator must be a &#x60;DECISION_JUDGE&#x60; evaluator. A list that mixes Decision Judge and other evaluator kinds returns 400; a list with only other evaluator kinds returns 422.  | 
**type** | **str** | Task type discriminator. Must be &#x60;\&quot;TEMPLATE_EVALUATION\&quot;&#x60;. | 
**run_configuration** | [**RunConfigurationRequest**](RunConfigurationRequest.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_task_request import CreateTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateTaskRequest from a JSON string
create_task_request_instance = CreateTaskRequest.from_json(json)
# print the JSON string representation of the object
print(CreateTaskRequest.to_json())

# convert the object into a dict
create_task_request_dict = create_task_request_instance.to_dict()
# create an instance of CreateTaskRequest from a dict
create_task_request_from_dict = CreateTaskRequest.from_dict(create_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


