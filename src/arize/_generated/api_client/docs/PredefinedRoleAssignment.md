# PredefinedRoleAssignment

A predefined space role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**SpaceRoleAssignmentType**](SpaceRoleAssignmentType.md) |  | 
**name** | [**UserSpaceRole**](UserSpaceRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.predefined_role_assignment import PredefinedRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of PredefinedRoleAssignment from a JSON string
predefined_role_assignment_instance = PredefinedRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(PredefinedRoleAssignment.to_json())

# convert the object into a dict
predefined_role_assignment_dict = predefined_role_assignment_instance.to_dict()
# create an instance of PredefinedRoleAssignment from a dict
predefined_role_assignment_from_dict = PredefinedRoleAssignment.from_dict(predefined_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


