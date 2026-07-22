# AgentRequestPreset

A named, reusable request payload bound to an agent integration. Embedded in `agent.config.request_presets[]`. There is no standalone preset endpoint; presets are managed exclusively via the nested collection on the parent integration.  Validation rules: - `name` is unique within the integration (case-sensitive). Length 1-255. - `description` length 0-1024 (nullable). - `config` must conform to the integration's `input_schema` *with   `required` dropped* (present fields validated; missing fields ignored). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Server-generated, opaque preset identifier. Read-only. | [optional] [readonly] 
**name** | **str** | Preset name (unique within the integration). Length 1-255. | 
**description** | **str** | Optional preset description (length 0-1024). | [optional] 
**config** | **Dict[str, object]** | Partial request body. Validated against the parent integration&#39;s &#x60;input_schema&#x60; with &#x60;required&#x60; dropped.  | 
**created_at** | **datetime** |  | [optional] [readonly] 
**updated_at** | **datetime** |  | [optional] [readonly] 

## Example

```python
from arize._generated.api_client.models.agent_request_preset import AgentRequestPreset

# TODO update the JSON string below
json = "{}"
# create an instance of AgentRequestPreset from a JSON string
agent_request_preset_instance = AgentRequestPreset.from_json(json)
# print the JSON string representation of the object
print(AgentRequestPreset.to_json())

# convert the object into a dict
agent_request_preset_dict = agent_request_preset_instance.to_dict()
# create an instance of AgentRequestPreset from a dict
agent_request_preset_from_dict = AgentRequestPreset.from_dict(agent_request_preset_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


