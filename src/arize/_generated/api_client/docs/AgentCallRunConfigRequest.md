# AgentCallRunConfigRequest

Strict request configuration for running an agent integration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_type** | **str** | Discriminator. Must be &#x60;\&quot;AGENT_CALL\&quot;&#x60;. | 
**integration_id** | **str** | Agent integration identifier (base64). The agent invoked for each dataset example. Must reference an integration of &#x60;type&#x60; &#x60;AGENT&#x60;; other integration types are rejected.  | 
**input_template** | **Dict[str, object]** | JSON request body sent to the agent for each dataset example. Must be a JSON object whose values conform to the agent integration&#39;s input schema. Mustache placeholders (&#x60;{{column}}&#x60;) are substituted with each dataset row&#39;s values before the request is sent.  | 

## Example

```python
from arize._generated.api_client.models.agent_call_run_config_request import AgentCallRunConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AgentCallRunConfigRequest from a JSON string
agent_call_run_config_request_instance = AgentCallRunConfigRequest.from_json(json)
# print the JSON string representation of the object
print(AgentCallRunConfigRequest.to_json())

# convert the object into a dict
agent_call_run_config_request_dict = agent_call_run_config_request_instance.to_dict()
# create an instance of AgentCallRunConfigRequest from a dict
agent_call_run_config_request_from_dict = AgentCallRunConfigRequest.from_dict(agent_call_run_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


