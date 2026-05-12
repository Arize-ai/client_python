# OrganizationMembershipInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**user_id** | **str** | The unique identifier of the user to add | 
**role** | [**OrganizationRoleAssignment**](OrganizationRoleAssignment.md) |  | 

## Example

```python
from arize._generated.api_client.models.organization_membership_input import OrganizationMembershipInput

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationMembershipInput from a JSON string
organization_membership_input_instance = OrganizationMembershipInput.from_json(json)
# print the JSON string representation of the object
print(OrganizationMembershipInput.to_json())

# convert the object into a dict
organization_membership_input_dict = organization_membership_input_instance.to_dict()
# create an instance of OrganizationMembershipInput from a dict
organization_membership_input_from_dict = OrganizationMembershipInput.from_dict(organization_membership_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


