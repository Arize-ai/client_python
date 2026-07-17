# CustomRoleAssignment

A custom RBAC role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**SpaceRoleAssignmentType**](SpaceRoleAssignmentType.md) | Discriminator identifying this as a custom RBAC role assignment. Always &#x60;CUSTOM&#x60; for this variant. | 
**id** | **str** | The unique identifier of the custom RBAC role. | 
**name** | **str** | Human-readable name of the custom role. Returned in responses only; ignored on input.  | [optional] [readonly] 

## Example

```python
from arize._generated.api_client.models.custom_role_assignment import CustomRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of CustomRoleAssignment from a JSON string
custom_role_assignment_instance = CustomRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(CustomRoleAssignment.to_json())

# convert the object into a dict
custom_role_assignment_dict = custom_role_assignment_instance.to_dict()
# create an instance of CustomRoleAssignment from a dict
custom_role_assignment_from_dict = CustomRoleAssignment.from_dict(custom_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


