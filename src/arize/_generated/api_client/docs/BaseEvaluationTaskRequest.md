# BaseEvaluationTaskRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Task name | 
**type** | **str** | Task type discriminator. Narrowed to a specific enum value by each concrete variant (&#x60;CreateTemplateEvaluationTaskRequest&#x60;, etc.).  | 
**project_id** | **str** | Project global ID (base64). Required when &#x60;dataset_id&#x60; is not provided. Mutually exclusive with &#x60;dataset_id&#x60;.  | [optional] 
**dataset_id** | **str** | Dataset global ID (base64). Required when &#x60;project_id&#x60; is not provided. Mutually exclusive with &#x60;project_id&#x60;.  | [optional] 
**experiment_ids** | **List[str]** | Experiment global IDs (base64). Required when &#x60;dataset_id&#x60; is provided (at least one entry). Must be omitted or empty for project-based tasks.  | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. Only supported on project-based tasks.  | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously. Only supported on project-based tasks. Must be &#x60;false&#x60; or omitted for dataset-based tasks.  | [optional] 
**query_filter** | **str** | Task-level query filter applied to all evaluated data. | [optional] 
**evaluators** | [**List[BaseEvaluationTaskRequestEvaluatorsInner]**](BaseEvaluationTaskRequestEvaluatorsInner.md) | Evaluators to attach (at least one required). | 

## Example

```python
from arize._generated.api_client.models.base_evaluation_task_request import BaseEvaluationTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of BaseEvaluationTaskRequest from a JSON string
base_evaluation_task_request_instance = BaseEvaluationTaskRequest.from_json(json)
# print the JSON string representation of the object
print(BaseEvaluationTaskRequest.to_json())

# convert the object into a dict
base_evaluation_task_request_dict = base_evaluation_task_request_instance.to_dict()
# create an instance of BaseEvaluationTaskRequest from a dict
base_evaluation_task_request_from_dict = BaseEvaluationTaskRequest.from_dict(base_evaluation_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


