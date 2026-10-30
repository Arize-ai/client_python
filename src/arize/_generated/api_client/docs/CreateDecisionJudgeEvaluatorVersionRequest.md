# CreateDecisionJudgeEvaluatorVersionRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**decision_judge_config** | [**DecisionJudgeConfigInput**](DecisionJudgeConfigInput.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_decision_judge_evaluator_version_request import CreateDecisionJudgeEvaluatorVersionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateDecisionJudgeEvaluatorVersionRequest from a JSON string
create_decision_judge_evaluator_version_request_instance = CreateDecisionJudgeEvaluatorVersionRequest.from_json(json)
# print the JSON string representation of the object
print(CreateDecisionJudgeEvaluatorVersionRequest.to_json())

# convert the object into a dict
create_decision_judge_evaluator_version_request_dict = create_decision_judge_evaluator_version_request_instance.to_dict()
# create an instance of CreateDecisionJudgeEvaluatorVersionRequest from a dict
create_decision_judge_evaluator_version_request_from_dict = CreateDecisionJudgeEvaluatorVersionRequest.from_dict(create_decision_judge_evaluator_version_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


