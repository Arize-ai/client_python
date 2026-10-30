# DecisionJudgeChoiceCriteriaInput

The options of a `CHOICE` question in a write request.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**options** | **Dict[str, str]** | Map of option label to a description of when the option applies. Between 2 and 255 options.  | 

## Example

```python
from arize._generated.api_client.models.decision_judge_choice_criteria_input import DecisionJudgeChoiceCriteriaInput

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeChoiceCriteriaInput from a JSON string
decision_judge_choice_criteria_input_instance = DecisionJudgeChoiceCriteriaInput.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeChoiceCriteriaInput.to_json())

# convert the object into a dict
decision_judge_choice_criteria_input_dict = decision_judge_choice_criteria_input_instance.to_dict()
# create an instance of DecisionJudgeChoiceCriteriaInput from a dict
decision_judge_choice_criteria_input_from_dict = DecisionJudgeChoiceCriteriaInput.from_dict(decision_judge_choice_criteria_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


