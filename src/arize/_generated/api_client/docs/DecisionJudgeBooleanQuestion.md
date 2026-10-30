# DecisionJudgeBooleanQuestion

A Decision Judge question with a yes or no answer.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Question name. Results are written as &#x60;eval.&lt;evaluator name&gt; &lt;question name&gt;.label&#x60; and &#x60;.score&#x60;.  | 
**type** | **str** | Discriminator identifying this as a boolean question. Always &#x60;BOOLEAN&#x60; for this variant. | 
**instructions** | **str** | The full instruction the model answers. The question name is not sent to the model. | 
**criteria** | [**DecisionJudgeBooleanCriteria**](DecisionJudgeBooleanCriteria.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.decision_judge_boolean_question import DecisionJudgeBooleanQuestion

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeBooleanQuestion from a JSON string
decision_judge_boolean_question_instance = DecisionJudgeBooleanQuestion.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeBooleanQuestion.to_json())

# convert the object into a dict
decision_judge_boolean_question_dict = decision_judge_boolean_question_instance.to_dict()
# create an instance of DecisionJudgeBooleanQuestion from a dict
decision_judge_boolean_question_from_dict = DecisionJudgeBooleanQuestion.from_dict(decision_judge_boolean_question_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


