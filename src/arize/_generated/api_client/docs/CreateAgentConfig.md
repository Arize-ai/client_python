# CreateAgentConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**endpoint** | **str** | HTTPS endpoint requests are sent to. Validated server-side and must resolve to a public address. | 
**headers** | **Dict[str, str]** | Cleartext header map. Encrypted at rest; never returned in responses. | [optional] 
**input_schema** | **Dict[str, object]** | JSON Schema (Draft-07) the endpoint&#39;s request body conforms to. | 
**request_presets** | [**List[CreateAgentRequestPresetInput]**](CreateAgentRequestPresetInput.md) | Optional initial presets for the integration. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_agent_config import CreateAgentConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAgentConfig from a JSON string
create_agent_config_instance = CreateAgentConfig.from_json(json)
# print the JSON string representation of the object
print(CreateAgentConfig.to_json())

# convert the object into a dict
create_agent_config_dict = create_agent_config_instance.to_dict()
# create an instance of CreateAgentConfig from a dict
create_agent_config_from_dict = CreateAgentConfig.from_dict(create_agent_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


