# OrganizationCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the organization (must be unique within the account) | 
**description** | **str** | A brief description of the organization&#39;s purpose. Defaults to an empty string if omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.organization_create import OrganizationCreate

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationCreate from a JSON string
organization_create_instance = OrganizationCreate.from_json(json)
# print the JSON string representation of the object
print(OrganizationCreate.to_json())

# convert the object into a dict
organization_create_dict = organization_create_instance.to_dict()
# create an instance of OrganizationCreate from a dict
organization_create_from_dict = OrganizationCreate.from_dict(organization_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


