# ApiKey

An API key as returned by `GET /v2/api-keys`. The `key_type` field discriminates the variant: - `USER` — standard user key; no bot user. - `SERVICE` — service key tied to a service account; includes a `bot_user` with the   service account's resolved role assignments. 

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
**bot_user** | [**ServiceKeyBotUser**](ServiceKeyBotUser.md) |  | 

## Example

```python
from arize._generated.api_client.models.api_key import ApiKey

# TODO update the JSON string below
json = "{}"
# create an instance of ApiKey from a JSON string
api_key_instance = ApiKey.from_json(json)
# print the JSON string representation of the object
print(ApiKey.to_json())

# convert the object into a dict
api_key_dict = api_key_instance.to_dict()
# create an instance of ApiKey from a dict
api_key_from_dict = ApiKey.from_dict(api_key_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


