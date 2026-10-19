# DefaultLLMIntegrationResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**integration_id** | **str** | The unique identifier of the integration. | 
**model_name** | **str** | The model name selected for the integration. | 

## Example

```python
from arize._generated.api_client.models.default_llm_integration_response import DefaultLLMIntegrationResponse

# TODO update the JSON string below
json = "{}"
# create an instance of DefaultLLMIntegrationResponse from a JSON string
default_llm_integration_response_instance = DefaultLLMIntegrationResponse.from_json(json)
# print the JSON string representation of the object
print(DefaultLLMIntegrationResponse.to_json())

# convert the object into a dict
default_llm_integration_response_dict = default_llm_integration_response_instance.to_dict()
# create an instance of DefaultLLMIntegrationResponse from a dict
default_llm_integration_response_from_dict = DefaultLLMIntegrationResponse.from_dict(default_llm_integration_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


