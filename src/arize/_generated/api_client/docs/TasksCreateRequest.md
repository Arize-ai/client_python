# TasksCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Task name | 
**type** | **str** | Task type discriminator. Must be &#x60;\&quot;template_evaluation\&quot;&#x60;. | 
**project_id** | **str** | Project global ID (base64). Required when &#x60;dataset_id&#x60; is not provided. Mutually exclusive with &#x60;dataset_id&#x60;.  | [optional] 
**dataset_id** | **str** | Dataset global ID (base64). Required for &#x60;run_experiment&#x60; tasks. | 
**experiment_ids** | **List[str]** | Experiment global IDs (base64). Required when &#x60;dataset_id&#x60; is provided (at least one entry). Must be omitted or empty for project-based tasks.  | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. Only supported on project-based tasks.  | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously. Only supported on project-based tasks. Must be &#x60;false&#x60; or omitted for dataset-based tasks.  | [optional] 
**query_filter** | **str** | Task-level query filter applied to all evaluated data. | [optional] 
**evaluators** | [**List[BaseEvaluationTaskRequestEvaluatorsInner]**](BaseEvaluationTaskRequestEvaluatorsInner.md) | Evaluators to attach (at least one required). | 
**run_configuration** | [**RunConfiguration**](RunConfiguration.md) |  | 

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


