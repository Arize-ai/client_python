# CreateUserRequest

User-level developer permissions are determined by the assigned account role.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Full name of the new user | 
**email** | **str** | Email address of the user to invite | 
**role** | [**UserRoleAssignmentRequest**](UserRoleAssignmentRequest.md) |  | 
**invite_mode** | [**InviteMode**](InviteMode.md) | Controls whether and how an invitation is sent | 

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


