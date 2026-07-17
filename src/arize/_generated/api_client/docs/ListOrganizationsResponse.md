# ListOrganizationsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organizations** | [**List[Organization]**](Organization.md) | A list of organizations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_organizations_response import ListOrganizationsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListOrganizationsResponse from a JSON string
list_organizations_response_instance = ListOrganizationsResponse.from_json(json)
# print the JSON string representation of the object
print(ListOrganizationsResponse.to_json())

# convert the object into a dict
list_organizations_response_dict = list_organizations_response_instance.to_dict()
# create an instance of ListOrganizationsResponse from a dict
list_organizations_response_from_dict = ListOrganizationsResponse.from_dict(list_organizations_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


