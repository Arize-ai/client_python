# CreateAgentRequestPresetInput

Write shape for an agent request preset on create. Server-generated fields (`id`, `created_at`, `updated_at`) are not accepted on input. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Preset name (unique within the integration). Length 1-255. | 
**description** | **str** | Optional preset description (length 0-1024). | [optional] 
**config** | **Dict[str, object]** | Partial request body. Validated against the parent integration&#39;s &#x60;input_schema&#x60; with &#x60;required&#x60; dropped.  | 

## Example

```python
from arize._generated.api_client.models.create_agent_request_preset_input import CreateAgentRequestPresetInput

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAgentRequestPresetInput from a JSON string
create_agent_request_preset_input_instance = CreateAgentRequestPresetInput.from_json(json)
# print the JSON string representation of the object
print(CreateAgentRequestPresetInput.to_json())

# convert the object into a dict
create_agent_request_preset_input_dict = create_agent_request_preset_input_instance.to_dict()
# create an instance of CreateAgentRequestPresetInput from a dict
create_agent_request_preset_input_from_dict = CreateAgentRequestPresetInput.from_dict(create_agent_request_preset_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


