# ApiKeysList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**api_keys** | [**List[ApiKey]**](ApiKey.md) | API keys owned by the authenticated user. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.api_keys_list200_response import ApiKeysList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ApiKeysList200Response from a JSON string
api_keys_list200_response_instance = ApiKeysList200Response.from_json(json)
# print the JSON string representation of the object
print(ApiKeysList200Response.to_json())

# convert the object into a dict
api_keys_list200_response_dict = api_keys_list200_response_instance.to_dict()
# create an instance of ApiKeysList200Response from a dict
api_keys_list200_response_from_dict = ApiKeysList200Response.from_dict(api_keys_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


