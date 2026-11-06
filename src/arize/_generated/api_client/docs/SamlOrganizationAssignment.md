# SamlOrganizationAssignment


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organization_id** | **str** | Organization ID in the caller account. | 
**role_id** | **str** | Role ID for this resource. | 

## Example

```python
from arize._generated.api_client.models.saml_organization_assignment import SamlOrganizationAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of SamlOrganizationAssignment from a JSON string
saml_organization_assignment_instance = SamlOrganizationAssignment.from_json(json)
# print the JSON string representation of the object
print(SamlOrganizationAssignment.to_json())

# convert the object into a dict
saml_organization_assignment_dict = saml_organization_assignment_instance.to_dict()
# create an instance of SamlOrganizationAssignment from a dict
saml_organization_assignment_from_dict = SamlOrganizationAssignment.from_dict(saml_organization_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


