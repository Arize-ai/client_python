# CreateApiKeyRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | User-defined name for the API key. | 
**description** | **str** | Optional user-defined description for the API key. | [optional] 
**key_type** | [**ApiKeyType**](ApiKeyType.md) | Type of the API key to create. Defaults to &#x60;USER&#x60;. - USER - Key that authenticates as the creating user with their full permissions.   &#x60;space_id&#x60; and &#x60;roles&#x60; must not be set (returns &#x60;400&#x60;). - SERVICE - Key scoped to a specific space backed by a dedicated bot user.   Requires &#x60;space_id&#x60;. All roles default to minimum privilege when omitted.  | [optional] 
**expires_at** | **datetime** | Optional expiration timestamp. If omitted the key never expires. | [optional] 
**space_id** | **str** | ID of the space this service key is scoped to. Required when &#x60;key_type&#x60; is &#x60;SERVICE&#x60;; invalid for &#x60;USER&#x60; keys (returns &#x60;400&#x60;).  | [optional] 
**roles** | [**ApiKeyRoles**](ApiKeyRoles.md) | Role assignments for the service key&#39;s bot user. Only valid when &#x60;key_type&#x60; is &#x60;SERVICE&#x60;; invalid for &#x60;USER&#x60; keys (returns &#x60;400&#x60;). When omitted, each role field defaults to minimum privilege: &#x60;space_role&#x60; → &#x60;MEMBER&#x60;, &#x60;org_role&#x60; → &#x60;READ_ONLY&#x60;, &#x60;account_role&#x60; → &#x60;MEMBER&#x60;.  | [optional] 

## Example

```python
from arize._generated.api_client.models.create_api_key_request import CreateApiKeyRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateApiKeyRequest from a JSON string
create_api_key_request_instance = CreateApiKeyRequest.from_json(json)
# print the JSON string representation of the object
print(CreateApiKeyRequest.to_json())

# convert the object into a dict
create_api_key_request_dict = create_api_key_request_instance.to_dict()
# create an instance of CreateApiKeyRequest from a dict
create_api_key_request_from_dict = CreateApiKeyRequest.from_dict(create_api_key_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


