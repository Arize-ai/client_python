# DecisionJudgeBooleanCriteria

Optional descriptions of what each answer means for a `BOOLEAN` question.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**true_means** | **str** | What a &#x60;true&#x60; answer means. | [optional] 
**false_means** | **str** | What a &#x60;false&#x60; answer means. | [optional] 

## Example

```python
from arize._generated.api_client.models.decision_judge_boolean_criteria import DecisionJudgeBooleanCriteria

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeBooleanCriteria from a JSON string
decision_judge_boolean_criteria_instance = DecisionJudgeBooleanCriteria.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeBooleanCriteria.to_json())

# convert the object into a dict
decision_judge_boolean_criteria_dict = decision_judge_boolean_criteria_instance.to_dict()
# create an instance of DecisionJudgeBooleanCriteria from a dict
decision_judge_boolean_criteria_from_dict = DecisionJudgeBooleanCriteria.from_dict(decision_judge_boolean_criteria_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


