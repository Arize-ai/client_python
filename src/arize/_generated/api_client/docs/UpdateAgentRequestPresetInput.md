# UpdateAgentRequestPresetInput

Write shape for an agent request preset on update. Matched by `name`: existing names update in place (preserving id/timestamps), new names insert, removed names delete. Server-generated fields (`id`, `created_at`, `updated_at`) are not accepted on input. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Preset name (unique within the integration). Length 1-255. | 
**description** | **str** | Optional preset description (length 0-1024). | [optional] 
**config** | **Dict[str, object]** | Partial request body. Validated against the parent integration&#39;s &#x60;input_schema&#x60; with &#x60;required&#x60; dropped.  | 

## Example

```python
from arize._generated.api_client.models.update_agent_request_preset_input import UpdateAgentRequestPresetInput

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateAgentRequestPresetInput from a JSON string
update_agent_request_preset_input_instance = UpdateAgentRequestPresetInput.from_json(json)
# print the JSON string representation of the object
print(UpdateAgentRequestPresetInput.to_json())

# convert the object into a dict
update_agent_request_preset_input_dict = update_agent_request_preset_input_instance.to_dict()
# create an instance of UpdateAgentRequestPresetInput from a dict
update_agent_request_preset_input_from_dict = UpdateAgentRequestPresetInput.from_dict(update_agent_request_preset_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


