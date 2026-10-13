# AgentCallRunConfig

Configuration for running an agent integration against each dataset example. The `input_template` is sent to the agent after Mustache substitution. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_type** | **str** | Discriminator. Must be &#x60;\&quot;AGENT_CALL\&quot;&#x60;. | 
**integration_id** | **str** | Agent integration identifier (base64). The agent invoked for each dataset example. Must reference an integration of &#x60;type&#x60; &#x60;AGENT&#x60;; other integration types are rejected.  | 
**input_template** | **Dict[str, object]** | JSON request body sent to the agent for each dataset example. Must be a JSON object whose values conform to the agent integration&#39;s input schema. Mustache placeholders (&#x60;{{column}}&#x60;) are substituted with each dataset row&#39;s values before the request is sent. The &#x60;dataset.&#x60; prefix is optional — &#x60;{{column}}&#x60; and &#x60;{{dataset.column}}&#x60; are equivalent, and responses (create, update, and read) always echo the normalized &#x60;{{column}}&#x60; form. Write every placeholder as a JSON string. A string that is exactly one placeholder (&#x60;\&quot;{{profile_id}}\&quot;&#x60;) is replaced by the row value cast to the type the input schema declares for that field: numeric strings become integers or numbers, &#x60;\&quot;true\&quot;&#x60;/&#x60;\&quot;false\&quot;&#x60; become booleans, JSON text becomes an object or array, and non-string values become strings. A placeholder embedded in other text (&#x60;\&quot;id-{{profile_id}}\&quot;&#x60;) is interpolated as text. A whole-value placeholder whose cell is null or whose column is missing is sent as &#x60;null&#x60;. Fields the schema does not type are sent with the dataset value&#39;s own type. If a value cannot be cast, that experiment run fails with an error naming the column, the value, and the target type; the other runs continue.  | 

## Example

```python
from arize._generated.api_client.models.agent_call_run_config import AgentCallRunConfig

# TODO update the JSON string below
json = "{}"
# create an instance of AgentCallRunConfig from a JSON string
agent_call_run_config_instance = AgentCallRunConfig.from_json(json)
# print the JSON string representation of the object
print(AgentCallRunConfig.to_json())

# convert the object into a dict
agent_call_run_config_dict = agent_call_run_config_instance.to_dict()
# create an instance of AgentCallRunConfig from a dict
agent_call_run_config_from_dict = AgentCallRunConfig.from_dict(agent_call_run_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


