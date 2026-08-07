# PredefinedRoleAssignmentRequest

A predefined space role assignment in a write request (strict form of PredefinedRoleAssignment).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**SpaceRoleAssignmentType**](SpaceRoleAssignmentType.md) | Discriminator identifying this as a predefined role assignment. Must be &#x60;PREDEFINED&#x60;. | 
**name** | [**UserSpaceRole**](UserSpaceRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.predefined_role_assignment_request import PredefinedRoleAssignmentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PredefinedRoleAssignmentRequest from a JSON string
predefined_role_assignment_request_instance = PredefinedRoleAssignmentRequest.from_json(json)
# print the JSON string representation of the object
print(PredefinedRoleAssignmentRequest.to_json())

# convert the object into a dict
predefined_role_assignment_request_dict = predefined_role_assignment_request_instance.to_dict()
# create an instance of PredefinedRoleAssignmentRequest from a dict
predefined_role_assignment_request_from_dict = PredefinedRoleAssignmentRequest.from_dict(predefined_role_assignment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


