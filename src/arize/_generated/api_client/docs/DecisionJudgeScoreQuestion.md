# DecisionJudgeScoreQuestion

A Decision Judge question whose answer is one level from an ordered scale.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Question name. Results are written as &#x60;eval.&lt;evaluator name&gt; &lt;question name&gt;.label&#x60; and &#x60;.score&#x60;.  | 
**type** | **str** | Discriminator identifying this as a score question. Always &#x60;SCORE&#x60; for this variant. | 
**instructions** | **str** | The full instruction the model answers. The question name is not sent to the model. | 
**criteria** | [**DecisionJudgeScoreCriteria**](DecisionJudgeScoreCriteria.md) |  | 
**direction** | [**OptimizationDirection**](OptimizationDirection.md) | Whether higher or lower levels are better. | 

## Example

```python
from arize._generated.api_client.models.decision_judge_score_question import DecisionJudgeScoreQuestion

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeScoreQuestion from a JSON string
decision_judge_score_question_instance = DecisionJudgeScoreQuestion.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeScoreQuestion.to_json())

# convert the object into a dict
decision_judge_score_question_dict = decision_judge_score_question_instance.to_dict()
# create an instance of DecisionJudgeScoreQuestion from a dict
decision_judge_score_question_from_dict = DecisionJudgeScoreQuestion.from_dict(decision_judge_score_question_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


