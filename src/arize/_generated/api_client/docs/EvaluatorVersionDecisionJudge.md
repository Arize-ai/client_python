# EvaluatorVersionDecisionJudge

Evaluator version carrying a Decision Judge configuration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for this version | 
**evaluator_id** | **str** | The parent evaluator ID | 
**commit_hash** | **str** | A unique hash identifying this version | 
**commit_message** | **str** | A message describing the changes in this version | 
**created_at** | **datetime** | When this version was created | 
**created_by_user_id** | **str** | The unique identifier for the user who created this version | 
**type** | **str** | Discriminator identifying this as a Decision Judge evaluator version. Always &#x60;DECISION_JUDGE&#x60; for this variant. | 
**decision_judge_config** | [**DecisionJudgeConfig**](DecisionJudgeConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_version_decision_judge import EvaluatorVersionDecisionJudge

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionDecisionJudge from a JSON string
evaluator_version_decision_judge_instance = EvaluatorVersionDecisionJudge.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionDecisionJudge.to_json())

# convert the object into a dict
evaluator_version_decision_judge_dict = evaluator_version_decision_judge_instance.to_dict()
# create an instance of EvaluatorVersionDecisionJudge from a dict
evaluator_version_decision_judge_from_dict = EvaluatorVersionDecisionJudge.from_dict(evaluator_version_decision_judge_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


