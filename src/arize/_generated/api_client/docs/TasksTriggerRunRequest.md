# TasksTriggerRunRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_start_time** | **datetime** | ISO 8601 start of the data window to evaluate. | [optional] 
**data_end_time** | **datetime** | ISO 8601 end of the data window to evaluate. If omitted, defaults to now. | [optional] 
**max_spans** | **int** | Maximum number of spans to process (default 10000). | [optional] 
**override_evaluations** | **bool** | Whether to re-evaluate data that already has evaluation labels (default false). | [optional] 
**experiment_ids** | **List[str]** | Experiment global IDs (base64) to run against. Only applicable for dataset-based tasks. | [optional] 

## Example

```python
from arize._generated.api_client.models.tasks_trigger_run_request import TasksTriggerRunRequest

# TODO update the JSON string below
json = "{}"
# create an instance of TasksTriggerRunRequest from a JSON string
tasks_trigger_run_request_instance = TasksTriggerRunRequest.from_json(json)
# print the JSON string representation of the object
print(TasksTriggerRunRequest.to_json())

# convert the object into a dict
tasks_trigger_run_request_dict = tasks_trigger_run_request_instance.to_dict()
# create an instance of TasksTriggerRunRequest from a dict
tasks_trigger_run_request_from_dict = TasksTriggerRunRequest.from_dict(tasks_trigger_run_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


