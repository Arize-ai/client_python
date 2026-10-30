# DecisionJudgeScoreCriteria

The ordered levels of a `SCORE` question.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**levels** | **List[str]** | Level descriptions, from the lowest level to the highest level. | 

## Example

```python
from arize._generated.api_client.models.decision_judge_score_criteria import DecisionJudgeScoreCriteria

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeScoreCriteria from a JSON string
decision_judge_score_criteria_instance = DecisionJudgeScoreCriteria.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeScoreCriteria.to_json())

# convert the object into a dict
decision_judge_score_criteria_dict = decision_judge_score_criteria_instance.to_dict()
# create an instance of DecisionJudgeScoreCriteria from a dict
decision_judge_score_criteria_from_dict = DecisionJudgeScoreCriteria.from_dict(decision_judge_score_criteria_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


