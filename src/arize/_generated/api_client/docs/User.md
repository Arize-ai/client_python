# User

An account user represents a member of the account. Users can be listed, updated, or removed from the account. Invitations in the acceptance flow appear as users with INVITED or EXPIRED status. Their IDs identify the invitation until acceptance; refresh the user list after acceptance to obtain the member ID. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**name** | **str** | Display name of the user. Pending invitations in the acceptance flow use the email address until acceptance. | 
**email** | **str** | An email address | 
**created_at** | **datetime** | Timestamp for when the user or invitation was created | 
**status** | [**UserStatus**](UserStatus.md) |  | 
**role** | [**UserRoleAssignment**](UserRoleAssignment.md) |  | 
**is_developer** | **bool** | Whether the user has developer permissions (can use the Arize API) | 

## Example

```python
from arize._generated.api_client.models.user import User

# TODO update the JSON string below
json = "{}"
# create an instance of User from a JSON string
user_instance = User.from_json(json)
# print the JSON string representation of the object
print(User.to_json())

# convert the object into a dict
user_dict = user_instance.to_dict()
# create an instance of User from a dict
user_from_dict = User.from_dict(user_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


