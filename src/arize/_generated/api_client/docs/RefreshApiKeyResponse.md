# RefreshApiKeyResponse

The refreshed API key credential and its metadata. Refresh replaces the key secret but preserves the key's identity (ID, name, type, bindings). Unlike key creation, refresh does **not** return `bot_user` details — refresh never creates a new service account and the existing bot user's bindings are unchanged. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the API key. | 
**name** | **str** | User-defined name for the API key. | 
**description** | **str** | Optional user-defined description for the API key. | [optional] 
**key_type** | [**ApiKeyType**](ApiKeyType.md) |  | 
**status** | [**ApiKeyStatus**](ApiKeyStatus.md) |  | 
**redacted_key** | **str** | Redacted version of the key suitable for display (e.g., \&quot;ak-abc...xyz\&quot;). | 
**created_at** | **datetime** | Timestamp when the key was created. | 
**expires_at** | **datetime** | Optional timestamp when the key will expire. | [optional] 
**created_by_user_id** | **str** | ID of the user who created the key. | 
**last_used_at** | **datetime** | Approximate timestamp when the key was last used for authentication. This value is periodically updated and may not reflect the most recent usage. | [optional] 
**key** | **str** | The full replacement API key value. **Only returned once** during refresh. Store it securely — it cannot be retrieved again.  | 

## Example

```python
from arize._generated.api_client.models.refresh_api_key_response import RefreshApiKeyResponse

# TODO update the JSON string below
json = "{}"
# create an instance of RefreshApiKeyResponse from a JSON string
refresh_api_key_response_instance = RefreshApiKeyResponse.from_json(json)
# print the JSON string representation of the object
print(RefreshApiKeyResponse.to_json())

# convert the object into a dict
refresh_api_key_response_dict = refresh_api_key_response_instance.to_dict()
# create an instance of RefreshApiKeyResponse from a dict
refresh_api_key_response_from_dict = RefreshApiKeyResponse.from_dict(refresh_api_key_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


