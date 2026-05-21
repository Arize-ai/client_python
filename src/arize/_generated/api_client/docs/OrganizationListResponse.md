# OrganizationListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organizations** | [**List[Organization]**](Organization.md) | A list of organizations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.organization_list_response import OrganizationListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationListResponse from a JSON string
organization_list_response_instance = OrganizationListResponse.from_json(json)
# print the JSON string representation of the object
print(OrganizationListResponse.to_json())

# convert the object into a dict
organization_list_response_dict = organization_list_response_instance.to_dict()
# create an instance of OrganizationListResponse from a dict
organization_list_response_from_dict = OrganizationListResponse.from_dict(organization_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


