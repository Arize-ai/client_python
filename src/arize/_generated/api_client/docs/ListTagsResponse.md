# ListTagsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tags** | [**List[Tag]**](Tag.md) | The tags attached to the resource, most recently updated first. Empty when the resource has no tags.  | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) | Pagination metadata. Tag lists are not paginated yet, so &#x60;has_more&#x60; is always &#x60;false&#x60; and &#x60;next_cursor&#x60; is always omitted. The field is present so that adding pagination later does not change the response shape.  | 

## Example

```python
from arize._generated.api_client.models.list_tags_response import ListTagsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListTagsResponse from a JSON string
list_tags_response_instance = ListTagsResponse.from_json(json)
# print the JSON string representation of the object
print(ListTagsResponse.to_json())

# convert the object into a dict
list_tags_response_dict = list_tags_response_instance.to_dict()
# create an instance of ListTagsResponse from a dict
list_tags_response_from_dict = ListTagsResponse.from_dict(list_tags_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


