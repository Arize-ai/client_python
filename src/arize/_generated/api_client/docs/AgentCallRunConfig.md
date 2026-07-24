# AgentCallRunConfig

Configuration for running an agent integration against each dataset example. The `input_template` is sent to the agent after Mustache substitution. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_type** | **str** | Discriminator. Must be &#x60;\&quot;AGENT_CALL\&quot;&#x60;. | 
**integration_id** | **str** | Agent integration identifier (base64). The agent invoked for each dataset example. Must reference an integration of &#x60;type&#x60; &#x60;AGENT&#x60;; other integration types are rejected.  | 
**input_template** | **Dict[str, object]** | JSON request body sent to the agent for each dataset example. Must be a JSON object whose values conform to the agent integration&#39;s input schema. Mustache placeholders (&#x60;{{column}}&#x60;) are substituted with each dataset row&#39;s values before the request is sent. The &#x60;dataset.&#x60; prefix is optional — &#x60;{{column}}&#x60; and &#x60;{{dataset.column}}&#x60; are equivalent, and responses (create, update, and read) always echo the normalized &#x60;{{column}}&#x60; form.  | 

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


