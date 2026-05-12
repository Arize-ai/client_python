# UserRoleAssignment

An account-level role assignment. Discriminated by `type`: - `predefined`: one of the predefined roles (`admin`, `member`, `annotator`) - `custom`: a custom RBAC role identified by its ID  Note: `custom` role assignments are not yet supported and are reserved for future use. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**UserRoleAssignmentType**](UserRoleAssignmentType.md) | Discriminator identifying this as a custom role assignment. Must be &#x60;custom&#x60;. | 
**name** | **str** | Human-readable name of the custom role. Returned in responses only; ignored on input.  | [readonly] 
**id** | **str** | The unique identifier of the custom RBAC role. | 

## Example

```python
from arize._generated.api_client.models.user_role_assignment import UserRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of UserRoleAssignment from a JSON string
user_role_assignment_instance = UserRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(UserRoleAssignment.to_json())

# convert the object into a dict
user_role_assignment_dict = user_role_assignment_instance.to_dict()
# create an instance of UserRoleAssignment from a dict
user_role_assignment_from_dict = UserRoleAssignment.from_dict(user_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


