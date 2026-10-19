# DefaultAgentRuntimeResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**integration_id** | **str** | The unique identifier of the integration. | 

## Example

```python
from arize._generated.api_client.models.default_agent_runtime_response import DefaultAgentRuntimeResponse

# TODO update the JSON string below
json = "{}"
# create an instance of DefaultAgentRuntimeResponse from a JSON string
default_agent_runtime_response_instance = DefaultAgentRuntimeResponse.from_json(json)
# print the JSON string representation of the object
print(DefaultAgentRuntimeResponse.to_json())

# convert the object into a dict
default_agent_runtime_response_dict = default_agent_runtime_response_instance.to_dict()
# create an instance of DefaultAgentRuntimeResponse from a dict
default_agent_runtime_response_from_dict = DefaultAgentRuntimeResponse.from_dict(default_agent_runtime_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


