# UpdateEvaluationTaskRequest

PATCH body for `template_evaluation` and `code_evaluation` tasks. The two types share the same updatable shape; the server derives the task type from the URL's task record. At least one field must be provided. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New task name. | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. Only applicable for project-based tasks. | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously. Only applicable for project-based tasks. | [optional] 
**query_filter** | **str** | Task-level query filter. Pass &#x60;null&#x60; to clear. | [optional] 
**evaluators** | [**List[BaseEvaluationTaskRequestEvaluatorsInner]**](BaseEvaluationTaskRequestEvaluatorsInner.md) | Replaces the entire evaluator list. At least one evaluator is required when provided. | [optional] 

## Example

```python
from arize._generated.api_client.models.update_evaluation_task_request import UpdateEvaluationTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateEvaluationTaskRequest from a JSON string
update_evaluation_task_request_instance = UpdateEvaluationTaskRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateEvaluationTaskRequest.to_json())

# convert the object into a dict
update_evaluation_task_request_dict = update_evaluation_task_request_instance.to_dict()
# create an instance of UpdateEvaluationTaskRequest from a dict
update_evaluation_task_request_from_dict = UpdateEvaluationTaskRequest.from_dict(update_evaluation_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


