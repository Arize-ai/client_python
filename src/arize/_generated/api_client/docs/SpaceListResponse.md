# SpaceListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**spaces** | [**List[Space]**](Space.md) | A list of spaces | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.space_list_response import SpaceListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of SpaceListResponse from a JSON string
space_list_response_instance = SpaceListResponse.from_json(json)
# print the JSON string representation of the object
print(SpaceListResponse.to_json())

# convert the object into a dict
space_list_response_dict = space_list_response_instance.to_dict()
# create an instance of SpaceListResponse from a dict
space_list_response_from_dict = SpaceListResponse.from_dict(space_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


