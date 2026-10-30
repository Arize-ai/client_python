# DecisionJudgeChoiceQuestion

A Decision Judge question whose answer is one option from a fixed set.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Question name. Results are written as &#x60;eval.&lt;evaluator name&gt; &lt;question name&gt;.label&#x60; and &#x60;.score&#x60;.  | 
**type** | **str** | Discriminator identifying this as a choice question. Always &#x60;CHOICE&#x60; for this variant. | 
**instructions** | **str** | The full instruction the model answers. The question name is not sent to the model. | 
**criteria** | [**DecisionJudgeChoiceCriteria**](DecisionJudgeChoiceCriteria.md) |  | 

## Example

```python
from arize._generated.api_client.models.decision_judge_choice_question import DecisionJudgeChoiceQuestion

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeChoiceQuestion from a JSON string
decision_judge_choice_question_instance = DecisionJudgeChoiceQuestion.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeChoiceQuestion.to_json())

# convert the object into a dict
decision_judge_choice_question_dict = decision_judge_choice_question_instance.to_dict()
# create an instance of DecisionJudgeChoiceQuestion from a dict
decision_judge_choice_question_from_dict = DecisionJudgeChoiceQuestion.from_dict(decision_judge_choice_question_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


