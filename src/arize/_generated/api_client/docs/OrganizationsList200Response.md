# OrganizationsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organizations** | [**List[Organization]**](Organization.md) | A list of organizations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.organizations_list200_response import OrganizationsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationsList200Response from a JSON string
organizations_list200_response_instance = OrganizationsList200Response.from_json(json)
# print the JSON string representation of the object
print(OrganizationsList200Response.to_json())

# convert the object into a dict
organizations_list200_response_dict = organizations_list200_response_instance.to_dict()
# create an instance of OrganizationsList200Response from a dict
organizations_list200_response_from_dict = OrganizationsList200Response.from_dict(organizations_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


