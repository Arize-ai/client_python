# DecisionJudgeBooleanQuestionInput

A Decision Judge question with a yes or no answer, in a write request.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Question name, unique within the evaluator (case-insensitive). The string &#x60;&lt;evaluator name&gt; &lt;question name&gt;&#x60; is the result column name and must match &#x60;^[a-zA-Z0-9_\\s\\-&amp;()]+$&#x60;. Leading and trailing whitespace is removed before the name is stored.  | 
**type** | **str** | Discriminator identifying this as a boolean question. Always &#x60;BOOLEAN&#x60; for this variant. | 
**instructions** | **str** | The full instruction the model answers. The question name is not sent to the model. Leading and trailing whitespace is removed before it is stored. | 
**criteria** | [**DecisionJudgeBooleanCriteriaInput**](DecisionJudgeBooleanCriteriaInput.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.decision_judge_boolean_question_input import DecisionJudgeBooleanQuestionInput

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeBooleanQuestionInput from a JSON string
decision_judge_boolean_question_input_instance = DecisionJudgeBooleanQuestionInput.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeBooleanQuestionInput.to_json())

# convert the object into a dict
decision_judge_boolean_question_input_dict = decision_judge_boolean_question_input_instance.to_dict()
# create an instance of DecisionJudgeBooleanQuestionInput from a dict
decision_judge_boolean_question_input_from_dict = DecisionJudgeBooleanQuestionInput.from_dict(decision_judge_boolean_question_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


