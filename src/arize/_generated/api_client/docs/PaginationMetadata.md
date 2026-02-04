# PaginationMetadata

Cursor-based pagination metadata. Use `next_cursor` in the subsequent request's `cursor` query parameter. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**next_cursor** | **str** | Opaque cursor for fetching the next page. Treat as an unreadable token. Present when &#x60;has_more&#x60; is true; omitted when &#x60;hasMore&#x60; is false.  | [optional] 
**has_more** | **bool** | True if another page of results is available. | 

## Example

```python
from arize._generated.api_client.models.pagination_metadata import PaginationMetadata

# TODO update the JSON string below
json = "{}"
# create an instance of PaginationMetadata from a JSON string
pagination_metadata_instance = PaginationMetadata.from_json(json)
# print the JSON string representation of the object
print(PaginationMetadata.to_json())

# convert the object into a dict
pagination_metadata_dict = pagination_metadata_instance.to_dict()
# create an instance of PaginationMetadata from a dict
pagination_metadata_from_dict = PaginationMetadata.from_dict(pagination_metadata_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


