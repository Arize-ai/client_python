# UpdateAgentConfig

Partial agent config for PATCH. All collection fields are replace-on-provide. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**endpoint** | **str** |  | [optional] 
**headers** | **Dict[str, str]** | Replace-on-provide. Pass &#x60;null&#x60; (or &#x60;{}&#x60;) to clear all headers. Encrypted at rest; never returned in responses.  | [optional] 
**input_schema** | **Dict[str, object]** | New JSON Schema for the request payload shape. | [optional] 
**request_presets** | [**List[UpdateAgentRequestPresetInput]**](UpdateAgentRequestPresetInput.md) | Replace-on-provide preset list, matched by &#x60;name&#x60;: existing names update in place (preserving id/timestamps), new names insert, removed names delete.  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_agent_config import UpdateAgentConfig

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateAgentConfig from a JSON string
update_agent_config_instance = UpdateAgentConfig.from_json(json)
# print the JSON string representation of the object
print(UpdateAgentConfig.to_json())

# convert the object into a dict
update_agent_config_dict = update_agent_config_instance.to_dict()
# create an instance of UpdateAgentConfig from a dict
update_agent_config_from_dict = UpdateAgentConfig.from_dict(update_agent_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


