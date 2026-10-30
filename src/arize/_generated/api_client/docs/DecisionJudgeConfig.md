# DecisionJudgeConfig

Configuration of a `DECISION_JUDGE` evaluator version. The model answers every question once per span, from the state that `state_template` renders. Each question writes its own result, named `eval.<evaluator name> <question name>.label` and `.score`. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ai_integration_id** | **str** | AI integration identifier (base64) of the integration that runs the evaluator. | 
**model_name** | **str** | The Decision Judge model the evaluator calls (e.g. &#x60;gpt-6-luna&#x60;). | 
**state_template** | **str** | Template of the state sent to the model, with &#x60;{variable}&#x60; placeholders filled from span attributes.  | 
**questions** | [**List[DecisionJudgeQuestion]**](DecisionJudgeQuestion.md) | The questions, in the order they were defined. | 
**data_granularity** | [**DataGranularity**](DataGranularity.md) | Data granularity level. Always &#x60;SPAN&#x60;; Decision Judge evaluators run on spans only. | [readonly] 

## Example

```python
from arize._generated.api_client.models.decision_judge_config import DecisionJudgeConfig

# TODO update the JSON string below
json = "{}"
# create an instance of DecisionJudgeConfig from a JSON string
decision_judge_config_instance = DecisionJudgeConfig.from_json(json)
# print the JSON string representation of the object
print(DecisionJudgeConfig.to_json())

# convert the object into a dict
decision_judge_config_dict = decision_judge_config_instance.to_dict()
# create an instance of DecisionJudgeConfig from a dict
decision_judge_config_from_dict = DecisionJudgeConfig.from_dict(decision_judge_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


