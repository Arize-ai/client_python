# SamlProjectAssignment


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | Project ID in the selected organization. | 
**role_id** | **str** | Role ID for this resource. | 

## Example

```python
from arize._generated.api_client.models.saml_project_assignment import SamlProjectAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of SamlProjectAssignment from a JSON string
saml_project_assignment_instance = SamlProjectAssignment.from_json(json)
# print the JSON string representation of the object
print(SamlProjectAssignment.to_json())

# convert the object into a dict
saml_project_assignment_dict = saml_project_assignment_instance.to_dict()
# create an instance of SamlProjectAssignment from a dict
saml_project_assignment_from_dict = SamlProjectAssignment.from_dict(saml_project_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


