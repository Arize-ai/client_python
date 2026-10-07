# UserApiKey


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the API key. | 
**name** | **str** | User-defined name for the API key. | 
**description** | **str** | Optional user-defined description for the API key. | [optional] 
**key_type** | **str** | Discriminator value for user keys. | 
**status** | [**ApiKeyStatus**](ApiKeyStatus.md) |  | 
**redacted_key** | **str** | Redacted version of the key suitable for display (e.g., \&quot;ak-abc...xyz\&quot;). | 
**created_at** | **datetime** | Timestamp when the key was created. | 
**expires_at** | **datetime** | Optional timestamp when the key will expire. | [optional] 
**created_by_user_id** | **str** | ID of the user who created the key. | 
**last_used_at** | **datetime** | Approximate timestamp when the key was last used for authentication. This value is periodically updated and may not reflect the most recent usage. | [optional] 

## Example

```python
from arize._generated.api_client.models.user_api_key import UserApiKey

# TODO update the JSON string below
json = "{}"
# create an instance of UserApiKey from a JSON string
user_api_key_instance = UserApiKey.from_json(json)
# print the JSON string representation of the object
print(UserApiKey.to_json())

# convert the object into a dict
user_api_key_dict = user_api_key_instance.to_dict()
# create an instance of UserApiKey from a dict
user_api_key_from_dict = UserApiKey.from_dict(user_api_key_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


