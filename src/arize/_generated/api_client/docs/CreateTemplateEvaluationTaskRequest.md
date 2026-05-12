# CreateTemplateEvaluationTaskRequest

Request body for creating a `template_evaluation` task. Requires `evaluators` and exactly one of `project_id` or `dataset_id`. When `dataset_id` is provided, `experiment_ids` must contain at least one entry. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Task name | 
**type** | **str** | Task type discriminator. Must be &#x60;\&quot;template_evaluation\&quot;&#x60;. | 
**project_id** | **str** | Project global ID (base64). Required when &#x60;dataset_id&#x60; is not provided. Mutually exclusive with &#x60;dataset_id&#x60;.  | [optional] 
**dataset_id** | **str** | Dataset global ID (base64). Required when &#x60;project_id&#x60; is not provided. Mutually exclusive with &#x60;project_id&#x60;.  | [optional] 
**experiment_ids** | **List[str]** | Experiment global IDs (base64). Required when &#x60;dataset_id&#x60; is provided (at least one entry). Must be omitted or empty for project-based tasks.  | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. Only supported on project-based tasks.  | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously. Only supported on project-based tasks. Must be &#x60;false&#x60; or omitted for dataset-based tasks.  | [optional] 
**query_filter** | **str** | Task-level query filter applied to all evaluated data. | [optional] 
**evaluators** | [**List[BaseEvaluationTaskRequestEvaluatorsInner]**](BaseEvaluationTaskRequestEvaluatorsInner.md) | Evaluators to attach (at least one required). | 

## Example

```python
from arize._generated.api_client.models.create_template_evaluation_task_request import CreateTemplateEvaluationTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateTemplateEvaluationTaskRequest from a JSON string
create_template_evaluation_task_request_instance = CreateTemplateEvaluationTaskRequest.from_json(json)
# print the JSON string representation of the object
print(CreateTemplateEvaluationTaskRequest.to_json())

# convert the object into a dict
create_template_evaluation_task_request_dict = create_template_evaluation_task_request_instance.to_dict()
# create an instance of CreateTemplateEvaluationTaskRequest from a dict
create_template_evaluation_task_request_from_dict = CreateTemplateEvaluationTaskRequest.from_dict(create_template_evaluation_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


