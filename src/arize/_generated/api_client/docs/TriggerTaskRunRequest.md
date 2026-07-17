# TriggerTaskRunRequest

Trigger body for `POST /v2/tasks/{task_id}/trigger`. The server derives the task type from the URL's task record and selects the appropriate schema; the body itself does not carry a `task_type` field.  | Task type | Schema | |---|---| | `TEMPLATE_EVALUATION` | `TriggerEvaluationTaskRunRequest` | | `CODE_EVALUATION` | `TriggerEvaluationTaskRunRequest` | | `RUN_EXPERIMENT` | `TriggerRunExperimentTaskRunRequest` |  Sending a field that is not valid for the resolved task type returns 400. For `TEMPLATE_EVALUATION` and `CODE_EVALUATION` tasks all trigger fields are optional — an empty body is valid and uses server defaults. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_start_time** | **datetime** | ISO 8601 start of the data window to evaluate. For model-based tasks, defaults to the task&#39;s last run time. Required on the first run (when no previous run exists). Not applicable to dataset-based tasks.  | [optional] 
**data_end_time** | **datetime** | ISO 8601 end of the data window to evaluate. For model-based tasks, defaults to now. Not applicable to dataset-based tasks.  | [optional] 
**max_spans** | **int** | Maximum number of spans to process (default 10000). | [optional] 
**override_evaluations** | **bool** | Whether to re-evaluate data that already has evaluation labels (default &#x60;false&#x60;).  | [optional] 
**experiment_ids** | **List[str]** | Experiment identifiers (base64) to run against. Only for dataset-based &#x60;TEMPLATE_EVALUATION&#x60; / &#x60;CODE_EVALUATION&#x60; tasks.  | [optional] 
**experiment_name** | **str** | Display name for the experiment to be created. Must be unique within the dataset.  | 
**dataset_version_id** | **str** | Dataset version identifier (base64). Defaults to the latest version when omitted.  | [optional] 
**example_ids** | **List[str]** | Specific example IDs to run against. Mutually exclusive with &#x60;max_examples&#x60;. When both are omitted, all examples are used.  | [optional] 
**max_examples** | **int** | Maximum number of examples to run (dataset order). Mutually exclusive with &#x60;example_ids&#x60;. When both are omitted, all examples are used.  | [optional] 
**tracing_metadata** | **Dict[str, str]** | Arbitrary key-value metadata. Providing this enables tracing for the run.  | [optional] 
**evaluation_task_ids** | **List[str]** | Task identifiers (base64) of evaluation tasks to trigger after the experiment run completes. Supported for all &#x60;RUN_EXPERIMENT&#x60; experiment types.  | [optional] 

## Example

```python
from arize._generated.api_client.models.trigger_task_run_request import TriggerTaskRunRequest

# TODO update the JSON string below
json = "{}"
# create an instance of TriggerTaskRunRequest from a JSON string
trigger_task_run_request_instance = TriggerTaskRunRequest.from_json(json)
# print the JSON string representation of the object
print(TriggerTaskRunRequest.to_json())

# convert the object into a dict
trigger_task_run_request_dict = trigger_task_run_request_instance.to_dict()
# create an instance of TriggerTaskRunRequest from a dict
trigger_task_run_request_from_dict = TriggerTaskRunRequest.from_dict(trigger_task_run_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


