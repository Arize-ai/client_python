# DecisionJudgeScoreCriteriaInput

The ordered levels of a `SCORE` question in a write request.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**levels** | **List[str]** | Level descriptions, from the lowest level to the highest level. Between 2 and 10 levels.  | 

## Example

```python
from arize._generated.api_client.models.decision_judge_score_criteria_input import DecisionJudgeScoreCriteriaInput

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeScoreCriteriaInput from a JSON string
decision_judge_score_criteria_input_instance = DecisionJudgeScoreCriteriaInput.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeScoreCriteriaInput.to_json())

# convert the object into a dict
decision_judge_score_criteria_input_dict = decision_judge_score_criteria_input_instance.to_dict()
# create an instance of DecisionJudgeScoreCriteriaInput from a dict
decision_judge_score_criteria_input_from_dict = DecisionJudgeScoreCriteriaInput.from_dict(decision_judge_score_criteria_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


