# RoleBinding


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the role binding. | 
**role_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**user_id** | **str** | ID of the user this binding assigns the role to.  For a **service key**, this is the ID of the key&#39;s bot user — not the ID of the person who created the key. Read it from &#x60;bot_user.id&#x60; on the &#x60;POST /v2/api-keys&#x60; response, or from &#x60;bot_user.id&#x60; on the matching service key entry returned by &#x60;GET /v2/api-keys&#x60;.  | 
**resource_type** | [**RoleBindingResourceType**](RoleBindingResourceType.md) |  | 
**resource_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**created_at** | **datetime** | Timestamp when the binding was created. | 
**updated_at** | **datetime** | Timestamp when the binding was last updated. | 

## Example

```python
from arize._generated.api_client.models.role_binding import RoleBinding

# TODO update the JSON string below
json = "{}"
# create an instance of RoleBinding from a JSON string
role_binding_instance = RoleBinding.from_json(json)
# print the JSON string representation of the object
print(RoleBinding.to_json())

# convert the object into a dict
role_binding_dict = role_binding_instance.to_dict()
# create an instance of RoleBinding from a dict
role_binding_from_dict = RoleBinding.from_dict(role_binding_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


