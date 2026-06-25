# SpanDeleteResponse

Result of a DELETE /v2/spans request.  `deleted_span_ids` lists every span ID confirmed deleted. `not_deleted_span_ids` lists every requested span ID that was **not** deleted.  `completed` indicates whether the server fully processed all data for the request — **not** whether all spans were found and deleted. A span may appear in `not_deleted_span_ids` even when `completed` is `true` if it was not found in the system (never ingested or already deleted).  When `completed` is `true`, every requested ID appears in exactly one of `deleted_span_ids` or `not_deleted_span_ids`. No retry is needed.  When `completed` is `false`, the server could not fully process all data. Some IDs in `not_deleted_span_ids` may still be deletable — retry the original full request to resolve them. The delete is idempotent. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**completed** | **bool** | &#x60;true&#x60; when the server fully processed all data for the request — both lists are complete and no retry is needed. &#x60;false&#x60; when processing could not fully complete; retry the original request. Note: &#x60;completed&#x60; reflects whether all data was processed, not whether all requested spans existed.  | 
**deleted_span_ids** | **List[str]** | Span IDs confirmed deleted in this request. | 
**not_deleted_span_ids** | **List[str]** | Requested span IDs that were not deleted. When &#x60;completed&#x60; is &#x60;true&#x60;, these were not found in the system (never ingested or already deleted). When &#x60;completed&#x60; is &#x60;false&#x60;, some IDs may not have been reached — retry to resolve them.  | 

## Example

```python
from arize._generated.api_client.models.span_delete_response import SpanDeleteResponse

# TODO update the JSON string below
json = "{}"
# create an instance of SpanDeleteResponse from a JSON string
span_delete_response_instance = SpanDeleteResponse.from_json(json)
# print the JSON string representation of the object
print(SpanDeleteResponse.to_json())

# convert the object into a dict
span_delete_response_dict = span_delete_response_instance.to_dict()
# create an instance of SpanDeleteResponse from a dict
span_delete_response_from_dict = SpanDeleteResponse.from_dict(span_delete_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


