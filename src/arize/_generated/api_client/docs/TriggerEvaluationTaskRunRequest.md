# TriggerEvaluationTaskRunRequest

Trigger request for `TEMPLATE_EVALUATION` or `CODE_EVALUATION` tasks. `data_start_time` and `data_end_time` together must span no more than 30 days. `data_start_time` must be before `data_end_time`. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_start_time** | **datetime** | ISO 8601 start of the data window to evaluate. For model-based tasks, defaults to the task&#39;s last run time. Required on the first run (when no previous run exists). Not applicable to dataset-based tasks.  | [optional] 
**data_end_time** | **datetime** | ISO 8601 end of the data window to evaluate. For model-based tasks, defaults to now. Not applicable to dataset-based tasks.  | [optional] 
**max_spans** | **int** | Maximum number of spans to process (default 10000). | [optional] 
**override_evaluations** | **bool** | Whether to re-evaluate data that already has evaluation labels (default &#x60;false&#x60;).  | [optional] 
**experiment_ids** | **List[str]** | Experiment identifiers (base64) to run against. Only for dataset-based &#x60;TEMPLATE_EVALUATION&#x60; / &#x60;CODE_EVALUATION&#x60; tasks.  | [optional] 

## Example

```python
from arize._generated.api_client.models.trigger_evaluation_task_run_request import TriggerEvaluationTaskRunRequest

# TODO update the JSON string below
json = "{}"
# create an instance of TriggerEvaluationTaskRunRequest from a JSON string
trigger_evaluation_task_run_request_instance = TriggerEvaluationTaskRunRequest.from_json(json)
# print the JSON string representation of the object
print(TriggerEvaluationTaskRunRequest.to_json())

# convert the object into a dict
trigger_evaluation_task_run_request_dict = trigger_evaluation_task_run_request_instance.to_dict()
# create an instance of TriggerEvaluationTaskRunRequest from a dict
trigger_evaluation_task_run_request_from_dict = TriggerEvaluationTaskRunRequest.from_dict(trigger_evaluation_task_run_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


