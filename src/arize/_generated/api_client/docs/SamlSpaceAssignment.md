# SamlSpaceAssignment


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**space_id** | **str** | Space ID in the selected organization. | 
**role_id** | **str** | Role ID for this resource. | 

## Example

```python
from arize._generated.api_client.models.saml_space_assignment import SamlSpaceAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of SamlSpaceAssignment from a JSON string
saml_space_assignment_instance = SamlSpaceAssignment.from_json(json)
# print the JSON string representation of the object
print(SamlSpaceAssignment.to_json())

# convert the object into a dict
saml_space_assignment_dict = saml_space_assignment_instance.to_dict()
# create an instance of SamlSpaceAssignment from a dict
saml_space_assignment_from_dict = SamlSpaceAssignment.from_dict(saml_space_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


