# OrganizationMembership


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the membership record | 
**user_id** | **str** | The unique identifier of the user | 
**organization_id** | **str** | The unique identifier of the organization | 
**role** | [**OrganizationRoleAssignment**](OrganizationRoleAssignment.md) |  | 

## Example

```python
from arize._generated.api_client.models.organization_membership import OrganizationMembership

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationMembership from a JSON string
organization_membership_instance = OrganizationMembership.from_json(json)
# print the JSON string representation of the object
print(OrganizationMembership.to_json())

# convert the object into a dict
organization_membership_dict = organization_membership_instance.to_dict()
# create an instance of OrganizationMembership from a dict
organization_membership_from_dict = OrganizationMembership.from_dict(organization_membership_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


