# ListResourceRestrictionsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_restrictions** | [**List[ResourceRestriction]**](ResourceRestriction.md) | A list of resource restriction records. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_resource_restrictions_response import ListResourceRestrictionsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListResourceRestrictionsResponse from a JSON string
list_resource_restrictions_response_instance = ListResourceRestrictionsResponse.from_json(json)
# print the JSON string representation of the object
print(ListResourceRestrictionsResponse.to_json())

# convert the object into a dict
list_resource_restrictions_response_dict = list_resource_restrictions_response_instance.to_dict()
# create an instance of ListResourceRestrictionsResponse from a dict
list_resource_restrictions_response_from_dict = ListResourceRestrictionsResponse.from_dict(list_resource_restrictions_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


