# CreateUserApiKeyRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**key_type** | **str** | Must be &#x60;\&quot;USER\&quot;&#x60;. | 
**name** | **str** | User-defined name for the API key. | 
**description** | **str** | Optional user-defined description for the API key. | [optional] 
**expires_at** | **datetime** | Optional expiration timestamp. If omitted the key never expires. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_user_api_key_request import CreateUserApiKeyRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateUserApiKeyRequest from a JSON string
create_user_api_key_request_instance = CreateUserApiKeyRequest.from_json(json)
# print the JSON string representation of the object
print(CreateUserApiKeyRequest.to_json())

# convert the object into a dict
create_user_api_key_request_dict = create_user_api_key_request_instance.to_dict()
# create an instance of CreateUserApiKeyRequest from a dict
create_user_api_key_request_from_dict = CreateUserApiKeyRequest.from_dict(create_user_api_key_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


