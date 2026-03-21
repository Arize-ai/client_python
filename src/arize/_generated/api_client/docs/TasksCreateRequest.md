# TasksCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Task name | 
**type** | **str** | Task type | 
**project_id** | **str** | Project global ID (base64). Required if dataset_id is not provided. Mutually exclusive with dataset_id. | [optional] 
**dataset_id** | **str** | Dataset global ID (base64). Required if project_id is not provided. Mutually exclusive with project_id. | [optional] 
**experiment_ids** | **List[str]** | Experiment global IDs (base64). Required when dataset_id is provided (at least one). Must be omitted or empty for project-based tasks. | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. Only supported on project tasks. | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously. Must be true or false for project-based tasks. Must be false or omitted for dataset-based tasks. | [optional] 
**query_filter** | **str** | Task-level query filter applied to all data. | [optional] 
**evaluators** | [**List[TasksCreateRequestEvaluatorsInner]**](TasksCreateRequestEvaluatorsInner.md) | Evaluators to attach (at least one required). | 

## Example

```python
from arize._generated.api_client.models.tasks_create_request import TasksCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of TasksCreateRequest from a JSON string
tasks_create_request_instance = TasksCreateRequest.from_json(json)
# print the JSON string representation of the object
print(TasksCreateRequest.to_json())

# convert the object into a dict
tasks_create_request_dict = tasks_create_request_instance.to_dict()
# create an instance of TasksCreateRequest from a dict
tasks_create_request_from_dict = TasksCreateRequest.from_dict(tasks_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


