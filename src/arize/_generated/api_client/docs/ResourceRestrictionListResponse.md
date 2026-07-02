# ResourceRestrictionListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_restrictions** | [**List[ResourceRestriction]**](ResourceRestriction.md) | A list of resource restriction records. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.resource_restriction_list_response import ResourceRestrictionListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ResourceRestrictionListResponse from a JSON string
resource_restriction_list_response_instance = ResourceRestrictionListResponse.from_json(json)
# print the JSON string representation of the object
print(ResourceRestrictionListResponse.to_json())

# convert the object into a dict
resource_restriction_list_response_dict = resource_restriction_list_response_instance.to_dict()
# create an instance of ResourceRestrictionListResponse from a dict
resource_restriction_list_response_from_dict = ResourceRestrictionListResponse.from_dict(resource_restriction_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


