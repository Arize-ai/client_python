# DecisionJudgeConfigInput

Configuration of a `DECISION_JUDGE` evaluator version in write requests. Decision Judge evaluators run on spans only. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ai_integration_id** | **str** | AI integration identifier (base64). Must reference an integration whose provider supports Decision Judge and that is available in the evaluator&#39;s space.  | 
**model_name** | **str** | The Decision Judge model to call. Must be a Decision Judge model of the integration&#39;s provider (e.g. &#x60;gpt-6-luna&#x60; for OpenAI).  | 
**state_template** | **str** | Template of the state sent to the model, with &#x60;{variable}&#x60; placeholders filled from span attributes. Leading and trailing whitespace is removed before it is stored.  | 
**questions** | [**List[DecisionJudgeQuestionInput]**](DecisionJudgeQuestionInput.md) | The questions to ask, between 1 and 64. | 

## Example

```python
from arize._generated.api_client.models.decision_judge_config_input import DecisionJudgeConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeConfigInput from a JSON string
decision_judge_config_input_instance = DecisionJudgeConfigInput.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeConfigInput.to_json())

# convert the object into a dict
decision_judge_config_input_dict = decision_judge_config_input_instance.to_dict()
# create an instance of DecisionJudgeConfigInput from a dict
decision_judge_config_input_from_dict = DecisionJudgeConfigInput.from_dict(decision_judge_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


