# BuiltinRoleAssignment

A builtin (predefined) space role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**SpaceRoleAssignmentType**](SpaceRoleAssignmentType.md) |  | 
**name** | [**UserSpaceRole**](UserSpaceRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.builtin_role_assignment import BuiltinRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of BuiltinRoleAssignment from a JSON string
builtin_role_assignment_instance = BuiltinRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(BuiltinRoleAssignment.to_json())

# convert the object into a dict
builtin_role_assignment_dict = builtin_role_assignment_instance.to_dict()
# create an instance of BuiltinRoleAssignment from a dict
builtin_role_assignment_from_dict = BuiltinRoleAssignment.from_dict(builtin_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


