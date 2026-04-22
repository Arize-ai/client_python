# TasksUpdateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New task name | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. Only applicable for project-based tasks. | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously. Only applicable for project-based tasks. | [optional] 
**query_filter** | **str** | Task-level query filter applied to all data. Pass null to clear. | [optional] 
**evaluators** | [**List[TasksCreateRequestEvaluatorsInner]**](TasksCreateRequestEvaluatorsInner.md) | Replaces the entire evaluator list. At least one evaluator is required when provided. | [optional] 

## Example

```python
from arize._generated.api_client.models.tasks_update_request import TasksUpdateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of TasksUpdateRequest from a JSON string
tasks_update_request_instance = TasksUpdateRequest.from_json(json)
# print the JSON string representation of the object
print(TasksUpdateRequest.to_json())

# convert the object into a dict
tasks_update_request_dict = tasks_update_request_instance.to_dict()
# create an instance of TasksUpdateRequest from a dict
tasks_update_request_from_dict = TasksUpdateRequest.from_dict(tasks_update_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


