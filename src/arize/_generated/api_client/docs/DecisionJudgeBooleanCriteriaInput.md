# DecisionJudgeBooleanCriteriaInput

Optional descriptions of what each answer means for a `BOOLEAN` question in a write request.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**true_means** | **str** | What a &#x60;true&#x60; answer means. | [optional] 
**false_means** | **str** | What a &#x60;false&#x60; answer means. | [optional] 

## Example

```python
from arize._generated.api_client.models.decision_judge_boolean_criteria_input import DecisionJudgeBooleanCriteriaInput

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeBooleanCriteriaInput from a JSON string
decision_judge_boolean_criteria_input_instance = DecisionJudgeBooleanCriteriaInput.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeBooleanCriteriaInput.to_json())

# convert the object into a dict
decision_judge_boolean_criteria_input_dict = decision_judge_boolean_criteria_input_instance.to_dict()
# create an instance of DecisionJudgeBooleanCriteriaInput from a dict
decision_judge_boolean_criteria_input_from_dict = DecisionJudgeBooleanCriteriaInput.from_dict(decision_judge_boolean_criteria_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


