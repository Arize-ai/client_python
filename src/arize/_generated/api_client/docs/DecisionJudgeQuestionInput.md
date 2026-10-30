# DecisionJudgeQuestionInput

One question of a Decision Judge evaluator in a write request. The `type` field selects the shape of `criteria`. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Question name, unique within the evaluator (case-insensitive). The string &#x60;&lt;evaluator name&gt; &lt;question name&gt;&#x60; is the result column name and must match &#x60;^[a-zA-Z0-9_\\s\\-&amp;()]+$&#x60;. Leading and trailing whitespace is removed before the name is stored.  | 
**type** | **str** | Discriminator identifying this as a boolean question. Always &#x60;BOOLEAN&#x60; for this variant. | 
**instructions** | **str** | The full instruction the model answers. The question name is not sent to the model. Leading and trailing whitespace is removed before it is stored. | 
**criteria** | [**DecisionJudgeScoreCriteriaInput**](DecisionJudgeScoreCriteriaInput.md) |  | 
**direction** | [**OptimizationDirection**](OptimizationDirection.md) | Whether higher or lower levels are better. Defaults to &#x60;MAXIMIZE&#x60; when omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.decision_judge_question_input import DecisionJudgeQuestionInput

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeQuestionInput from a JSON string
decision_judge_question_input_instance = DecisionJudgeQuestionInput.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeQuestionInput.to_json())

# convert the object into a dict
decision_judge_question_input_dict = decision_judge_question_input_instance.to_dict()
# create an instance of DecisionJudgeQuestionInput from a dict
decision_judge_question_input_from_dict = DecisionJudgeQuestionInput.from_dict(decision_judge_question_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


