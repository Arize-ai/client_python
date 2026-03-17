# ApiKeyCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | User-defined name for the API key. | 
**description** | **str** | Optional user-defined description for the API key. | [optional] 
**key_type** | **str** | Type of the API key to create. Defaults to &#x60;user&#x60;. - user - Key that authenticates as the creating user with their full permissions.   &#x60;space_id&#x60; and &#x60;roles&#x60; must not be set (returns &#x60;400&#x60;). - service - Key scoped to a specific space backed by a dedicated bot user.   Requires &#x60;space_id&#x60;. All roles default to minimum privilege when omitted.  | [optional] [default to 'user']
**expires_at** | **datetime** | Optional expiration timestamp. If omitted the key never expires. | [optional] 
**space_id** | **str** | ID of the space this service key is scoped to. Required when &#x60;key_type&#x60; is &#x60;service&#x60;; invalid for &#x60;user&#x60; keys (returns &#x60;400&#x60;).  | [optional] 
**roles** | [**ApiKeyRoles**](ApiKeyRoles.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.api_key_create import ApiKeyCreate

# TODO update the JSON string below
json = "{}"
# create an instance of ApiKeyCreate from a JSON string
api_key_create_instance = ApiKeyCreate.from_json(json)
# print the JSON string representation of the object
print(ApiKeyCreate.to_json())

# convert the object into a dict
api_key_create_dict = api_key_create_instance.to_dict()
# create an instance of ApiKeyCreate from a dict
api_key_create_from_dict = ApiKeyCreate.from_dict(api_key_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


