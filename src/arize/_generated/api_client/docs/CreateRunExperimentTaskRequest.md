# CreateRunExperimentTaskRequest

Request body for creating a `run_experiment` task. Requires `dataset_id` and `run_configuration`. Does not support continuous execution — runs are triggered explicitly via `POST /v2/tasks/{task_id}/trigger`. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Task name | 
**type** | **str** | Task type discriminator. Must be &#x60;\&quot;run_experiment\&quot;&#x60;. | 
**dataset_id** | **str** | Dataset global ID (base64). Required for &#x60;run_experiment&#x60; tasks. | 
**run_configuration** | [**RunConfiguration**](RunConfiguration.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_run_experiment_task_request import CreateRunExperimentTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateRunExperimentTaskRequest from a JSON string
create_run_experiment_task_request_instance = CreateRunExperimentTaskRequest.from_json(json)
# print the JSON string representation of the object
print(CreateRunExperimentTaskRequest.to_json())

# convert the object into a dict
create_run_experiment_task_request_dict = create_run_experiment_task_request_instance.to_dict()
# create an instance of CreateRunExperimentTaskRequest from a dict
create_run_experiment_task_request_from_dict = CreateRunExperimentTaskRequest.from_dict(create_run_experiment_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


