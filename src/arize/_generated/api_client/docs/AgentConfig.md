# AgentConfig

Configuration for `type: AGENT` integrations: a customer-hosted HTTPS endpoint plus a JSON Schema describing the request payload. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**endpoint** | **str** | HTTPS endpoint URL Arize calls for replay. Validated server-side for SSRF (must resolve to a public address).  | 
**has_headers** | **bool** | Whether any headers are configured. Read-only — derived from &#x60;headers&#x60; on write. Header values are never returned.  | [readonly] 
**input_schema** | **Dict[str, object]** | JSON Schema (Draft-07) the endpoint&#39;s request body conforms to.  | 
**request_presets** | [**List[AgentRequestPreset]**](AgentRequestPreset.md) | Named, reusable request payloads. Replace-on-provide on PATCH. Always present; an integration with no presets returns &#x60;[]&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.agent_config import AgentConfig

# TODO update the JSON string below
json = "{}"
# create an instance of AgentConfig from a JSON string
agent_config_instance = AgentConfig.from_json(json)
# print the JSON string representation of the object
print(AgentConfig.to_json())

# convert the object into a dict
agent_config_dict = agent_config_instance.to_dict()
# create an instance of AgentConfig from a dict
agent_config_from_dict = AgentConfig.from_dict(agent_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


