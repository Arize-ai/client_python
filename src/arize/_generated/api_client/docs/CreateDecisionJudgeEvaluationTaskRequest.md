# CreateDecisionJudgeEvaluationTaskRequest

Request body for creating a `DECISION_JUDGE_EVALUATION` task. Requires `project_id` and at least one Decision Judge evaluator. The task evaluates single spans, so only the span shape (`query_filter`, per-evaluator `column_mappings`/`query_filter`) is accepted. `dataset_id`, `experiment_ids`, `query_filters` and `query_mappings` return 422. Returns 403 when Decision Judge evaluators are not enabled for the account. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Task name | 
**project_id** | **str** | Project identifier (base64). Decision Judge tasks run on the spans of a project. Datasets are not supported.  | 
**sampling_rate** | **float** | Sampling rate between 0 and 1. | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously on incoming spans. | [optional] 
**query_filter** | **str** | Task-level query filter applied to all evaluated spans. | [optional] 
**evaluators** | [**List[SpanEvaluatorInput]**](SpanEvaluatorInput.md) | Decision Judge evaluators to attach (at least one required). Every evaluator must be a &#x60;DECISION_JUDGE&#x60; evaluator. A list that mixes Decision Judge and other evaluator kinds returns 400; a list with only other evaluator kinds returns 422.  | 
**type** | **str** | Task type discriminator. Must be &#x60;\&quot;DECISION_JUDGE_EVALUATION\&quot;&#x60;. | 

## Example

```python
from arize._generated.api_client.models.create_decision_judge_evaluation_task_request import CreateDecisionJudgeEvaluationTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateDecisionJudgeEvaluationTaskRequest from a JSON string
create_decision_judge_evaluation_task_request_instance = CreateDecisionJudgeEvaluationTaskRequest.from_json(json)
# print the JSON string representation of the object
print(CreateDecisionJudgeEvaluationTaskRequest.to_json())

# convert the object into a dict
create_decision_judge_evaluation_task_request_dict = create_decision_judge_evaluation_task_request_instance.to_dict()
# create an instance of CreateDecisionJudgeEvaluationTaskRequest from a dict
create_decision_judge_evaluation_task_request_from_dict = CreateDecisionJudgeEvaluationTaskRequest.from_dict(create_decision_judge_evaluation_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


