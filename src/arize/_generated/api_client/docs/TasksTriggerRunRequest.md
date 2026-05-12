# TasksTriggerRunRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_start_time** | **datetime** | ISO 8601 start of the data window to evaluate. If omitted, defaults to the task&#39;s last run time (or 7 days ago on first run).  | [optional] 
**data_end_time** | **datetime** | ISO 8601 end of the data window to evaluate. If omitted, defaults to now.  | [optional] 
**max_spans** | **int** | Maximum number of spans to process (default 10000). | [optional] 
**override_evaluations** | **bool** | Whether to re-evaluate data that already has evaluation labels (default &#x60;false&#x60;).  | [optional] 
**experiment_ids** | **List[str]** | Experiment global IDs (base64) to run against. Only for dataset-based &#x60;template_evaluation&#x60; / &#x60;code_evaluation&#x60; tasks.  | [optional] 
**experiment_name** | **str** | Display name for the experiment to be created. Must be unique within the dataset.  | 
**dataset_version_id** | **str** | Dataset version global ID (base64). Defaults to the latest version when omitted.  | [optional] 
**example_ids** | **List[str]** | Specific example IDs to run against. Mutually exclusive with &#x60;max_examples&#x60;. When both are omitted, all examples are used.  | [optional] 
**max_examples** | **int** | Maximum number of examples to run (dataset order). Mutually exclusive with &#x60;example_ids&#x60;. When both are omitted, all examples are used.  | [optional] 
**tracing_metadata** | **Dict[str, str]** | Arbitrary key-value metadata. Providing this enables tracing for the run.  | [optional] 
**evaluation_task_ids** | **List[str]** | Task global IDs (base64) of evaluation tasks to trigger after the experiment run completes. Supported for all &#x60;run_experiment&#x60; experiment types.  | [optional] 

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


