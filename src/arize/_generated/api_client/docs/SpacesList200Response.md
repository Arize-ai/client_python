# SpacesList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**spaces** | [**List[Space]**](Space.md) | A list of spaces | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.spaces_list200_response import SpacesList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of SpacesList200Response from a JSON string
spaces_list200_response_instance = SpacesList200Response.from_json(json)
# print the JSON string representation of the object
print(SpacesList200Response.to_json())

# convert the object into a dict
spaces_list200_response_dict = spaces_list200_response_instance.to_dict()
# create an instance of SpacesList200Response from a dict
spaces_list200_response_from_dict = SpacesList200Response.from_dict(spaces_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


