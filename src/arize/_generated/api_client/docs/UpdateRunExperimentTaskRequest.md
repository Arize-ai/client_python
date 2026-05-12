# UpdateRunExperimentTaskRequest

PATCH body for `run_experiment` tasks. The server derives the task type from the URL's task record. At least one of `name` or `run_configuration` must be provided. When `run_configuration` is provided the stored config is fully replaced (existing configs are marked inactive and the new config is inserted atomically in a transaction). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New task name. | [optional] 
**run_configuration** | [**RunConfiguration**](RunConfiguration.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_run_experiment_task_request import UpdateRunExperimentTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateRunExperimentTaskRequest from a JSON string
update_run_experiment_task_request_instance = UpdateRunExperimentTaskRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateRunExperimentTaskRequest.to_json())

# convert the object into a dict
update_run_experiment_task_request_dict = update_run_experiment_task_request_instance.to_dict()
# create an instance of UpdateRunExperimentTaskRequest from a dict
update_run_experiment_task_request_from_dict = UpdateRunExperimentTaskRequest.from_dict(update_run_experiment_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


