# TriggerRunExperimentTaskRunRequest

Trigger request for `run_experiment` tasks. `example_ids` and `max_examples` are mutually exclusive; at most one may be provided. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_name** | **str** | Display name for the experiment to be created. Must be unique within the dataset.  | 
**dataset_version_id** | **str** | Dataset version global ID (base64). Defaults to the latest version when omitted.  | [optional] 
**example_ids** | **List[str]** | Specific example IDs to run against. Mutually exclusive with &#x60;max_examples&#x60;. When both are omitted, all examples are used.  | [optional] 
**max_examples** | **int** | Maximum number of examples to run (dataset order). Mutually exclusive with &#x60;example_ids&#x60;. When both are omitted, all examples are used.  | [optional] 
**tracing_metadata** | **Dict[str, str]** | Arbitrary key-value metadata. Providing this enables tracing for the run.  | [optional] 
**evaluation_task_ids** | **List[str]** | Task global IDs (base64) of evaluation tasks to trigger after the experiment run completes. Supported for all &#x60;run_experiment&#x60; experiment types.  | [optional] 

## Example

```python
from arize._generated.api_client.models.trigger_run_experiment_task_run_request import TriggerRunExperimentTaskRunRequest

# TODO update the JSON string below
json = "{}"
# create an instance of TriggerRunExperimentTaskRunRequest from a JSON string
trigger_run_experiment_task_run_request_instance = TriggerRunExperimentTaskRunRequest.from_json(json)
# print the JSON string representation of the object
print(TriggerRunExperimentTaskRunRequest.to_json())

# convert the object into a dict
trigger_run_experiment_task_run_request_dict = trigger_run_experiment_task_run_request_instance.to_dict()
# create an instance of TriggerRunExperimentTaskRunRequest from a dict
trigger_run_experiment_task_run_request_from_dict = TriggerRunExperimentTaskRunRequest.from_dict(trigger_run_experiment_task_run_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


