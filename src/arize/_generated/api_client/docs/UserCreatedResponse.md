# UserCreatedResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the user | 
**name** | **str** | Full name of the user | 
**email** | **str** | Email address of the user | 
**role** | [**UserRole**](UserRole.md) | Account-level role assigned to the user | 
**created_at** | **datetime** | Timestamp for when the user was created | 
**status** | [**UserStatus**](UserStatus.md) |  | 
**is_developer** | **bool** | Whether the user has developer permissions (can create GraphQL API keys) | 
**invite_mode** | [**InviteMode**](InviteMode.md) | The invite mode used when the user was created. | 
**temporary_password** | **str** | Temporary password issued when &#x60;invite_mode&#x60; is &#x60;temporary_password&#x60;. Only present in the &#x60;POST /v2/users&#x60; 201 Created response.  | [optional] 

## Example

```python
from arize._generated.api_client.models.user_created_response import UserCreatedResponse

# TODO update the JSON string below
json = "{}"
# create an instance of UserCreatedResponse from a JSON string
user_created_response_instance = UserCreatedResponse.from_json(json)
# print the JSON string representation of the object
print(UserCreatedResponse.to_json())

# convert the object into a dict
user_created_response_dict = user_created_response_instance.to_dict()
# create an instance of UserCreatedResponse from a dict
user_created_response_from_dict = UserCreatedResponse.from_dict(user_created_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


