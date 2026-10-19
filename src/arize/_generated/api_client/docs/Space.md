# Space

A space is a container within an organization for grouping related projects, datasets, and experiments. Spaces enable team collaboration or isolated experimentation with role-based access control. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the space | 
**organization_id** | **str** | The unique identifier of the organization that contains the space. Older server versions may omit this field. | [optional] 
**name** | **str** | Name of the space | 
**description** | **str** | A brief description of the space&#39;s purpose | 
**created_at** | **datetime** | Timestamp for when the space was created | 
**is_private** | **bool** | Whether the space is private. Private spaces are only visible to their members and account/org/space admins.  | 
**default_llm_integration** | [**DefaultLLMIntegrationResponse**](DefaultLLMIntegrationResponse.md) |  | [optional] 
**ml_models_enabled** | **bool** | Whether machine learning models are enabled for the space. | [optional] 
**gradient_start_color** | **str** | The first color in the gradient used for the space&#39;s avatar. | [optional] 
**gradient_end_color** | **str** | The second color in the gradient used for the space&#39;s avatar. | [optional] 
**default_agent_runtime** | [**DefaultAgentRuntimeResponse**](DefaultAgentRuntimeResponse.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.space import Space

# TODO update the JSON string below
json = "{}"
# create an instance of Space from a JSON string
space_instance = Space.from_json(json)
# print the JSON string representation of the object
print(Space.to_json())

# convert the object into a dict
space_dict = space_instance.to_dict()
# create an instance of Space from a dict
space_from_dict = Space.from_dict(space_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


