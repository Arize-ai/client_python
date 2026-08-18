# CreatedServiceApiKey


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the API key. | 
**name** | **str** | User-defined name for the API key. | 
**description** | **str** | Optional user-defined description for the API key. | [optional] 
**key_type** | **str** | Discriminator value for service keys. | 
**status** | [**ApiKeyStatus**](ApiKeyStatus.md) |  | 
**redacted_key** | **str** | Redacted version of the key suitable for display (e.g., \&quot;ak-abc...xyz\&quot;). | 
**created_at** | **datetime** | Timestamp when the key was created. | 
**expires_at** | **datetime** | Optional timestamp when the key will expire. | [optional] 
**created_by_user_id** | **str** | ID of the user who created the key. | 
**last_used_at** | **datetime** | Approximate timestamp when the key was last used for authentication. This value is periodically updated and may not reflect the most recent usage. | [optional] 
**key** | **str** | The full API key value. **Only returned once** at creation or refresh time. Store it securely — it cannot be retrieved again.  | 
**bot_user** | [**ServiceKeyBotUser**](ServiceKeyBotUser.md) |  | 

## Example

```python
from arize._generated.api_client.models.created_service_api_key import CreatedServiceApiKey

# TODO update the JSON string below
json = "{}"
# create an instance of CreatedServiceApiKey from a JSON string
created_service_api_key_instance = CreatedServiceApiKey.from_json(json)
# print the JSON string representation of the object
print(CreatedServiceApiKey.to_json())

# convert the object into a dict
created_service_api_key_dict = created_service_api_key_instance.to_dict()
# create an instance of CreatedServiceApiKey from a dict
created_service_api_key_from_dict = CreatedServiceApiKey.from_dict(created_service_api_key_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


