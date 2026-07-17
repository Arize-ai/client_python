# UpdateTaskRequest

PATCH body for `PATCH /v2/tasks/{task_id}`. The server derives the task type from the URL's task record and selects the appropriate schema; the body itself does not carry a `type` field.  | Task type | Schema | |---|---| | `TEMPLATE_EVALUATION` | `UpdateEvaluationTaskRequest` | | `CODE_EVALUATION` | `UpdateEvaluationTaskRequest` | | `RUN_EXPERIMENT` | `UpdateRunExperimentTaskRequest` |  Sending a field that is not valid for the resolved task type returns 400 (e.g. `evaluators` on a `RUN_EXPERIMENT` task, or `run_configuration` on an evaluation task). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New task name. | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. Only applicable for project-based tasks. | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously. Only applicable for project-based tasks. | [optional] 
**query_filter** | **str** | Task-level query filter. Pass &#x60;null&#x60; to clear. | [optional] 
**evaluators** | [**List[TaskEvaluatorInput]**](TaskEvaluatorInput.md) | Replaces the entire evaluator list. At least one evaluator is required when provided. | [optional] 
**run_configuration** | [**RunConfiguration**](RunConfiguration.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_task_request import UpdateTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateTaskRequest from a JSON string
update_task_request_instance = UpdateTaskRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateTaskRequest.to_json())

# convert the object into a dict
update_task_request_dict = update_task_request_instance.to_dict()
# create an instance of UpdateTaskRequest from a dict
update_task_request_from_dict = UpdateTaskRequest.from_dict(update_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


