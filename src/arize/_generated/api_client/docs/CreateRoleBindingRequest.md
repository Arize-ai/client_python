# CreateRoleBindingRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**user_id** | **str** | ID of the user to bind the role to.  For a **service key**, this is the ID of the key&#39;s bot user — not the ID of the person who created the key. Read it from &#x60;bot_user.id&#x60; on the &#x60;POST /v2/api-keys&#x60; response, or from &#x60;bot_user.id&#x60; on the matching service key entry returned by &#x60;GET /v2/api-keys&#x60;.  | 
**resource_type** | [**RoleBindingResourceType**](RoleBindingResourceType.md) |  | 
**resource_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 

## Example

```python
from arize._generated.api_client.models.create_role_binding_request import CreateRoleBindingRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateRoleBindingRequest from a JSON string
create_role_binding_request_instance = CreateRoleBindingRequest.from_json(json)
# print the JSON string representation of the object
print(CreateRoleBindingRequest.to_json())

# convert the object into a dict
create_role_binding_request_dict = create_role_binding_request_instance.to_dict()
# create an instance of CreateRoleBindingRequest from a dict
create_role_binding_request_from_dict = CreateRoleBindingRequest.from_dict(create_role_binding_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


