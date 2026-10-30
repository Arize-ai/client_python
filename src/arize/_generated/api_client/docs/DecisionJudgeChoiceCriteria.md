# DecisionJudgeChoiceCriteria

The options of a `CHOICE` question.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**options** | **Dict[str, str]** | Map of option label to a description of when the option applies. | 

## Example

```python
from arize._generated.api_client.models.decision_judge_choice_criteria import DecisionJudgeChoiceCriteria

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeChoiceCriteria from a JSON string
decision_judge_choice_criteria_instance = DecisionJudgeChoiceCriteria.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeChoiceCriteria.to_json())

# convert the object into a dict
decision_judge_choice_criteria_dict = decision_judge_choice_criteria_instance.to_dict()
# create an instance of DecisionJudgeChoiceCriteria from a dict
decision_judge_choice_criteria_from_dict = DecisionJudgeChoiceCriteria.from_dict(decision_judge_choice_criteria_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


