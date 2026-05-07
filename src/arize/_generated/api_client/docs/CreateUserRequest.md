# CreateUserRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Full name of the new user | 
**email** | **str** | Email address of the user to invite | 
**role** | [**UserRole**](UserRole.md) | Account-level role to assign to the user | 
**invite_mode** | [**InviteMode**](InviteMode.md) | Controls whether and how an invitation is sent | 
**is_developer** | **bool** | Whether the user should have developer permissions (can create GraphQL API keys). Defaults to &#x60;true&#x60; for &#x60;admin&#x60; and &#x60;member&#x60; roles, and &#x60;false&#x60; for &#x60;annotator&#x60;.  | [optional] 

## Example

```python
from arize._generated.api_client.models.create_user_request import CreateUserRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateUserRequest from a JSON string
create_user_request_instance = CreateUserRequest.from_json(json)
# print the JSON string representation of the object
print(CreateUserRequest.to_json())

# convert the object into a dict
create_user_request_dict = create_user_request_instance.to_dict()
# create an instance of CreateUserRequest from a dict
create_user_request_from_dict = CreateUserRequest.from_dict(create_user_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


