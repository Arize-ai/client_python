# SpanDeletePartialResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**deleted_span_ids** | **List[str]** | Span IDs confirmed deleted across all successfully processed intervals. | 

## Example

```python
from arize._generated.api_client.models.span_delete_partial_response import SpanDeletePartialResponse

# TODO update the JSON string below
json = "{}"
# create an instance of SpanDeletePartialResponse from a JSON string
span_delete_partial_response_instance = SpanDeletePartialResponse.from_json(json)
# print the JSON string representation of the object
print(SpanDeletePartialResponse.to_json())

# convert the object into a dict
span_delete_partial_response_dict = span_delete_partial_response_instance.to_dict()
# create an instance of SpanDeletePartialResponse from a dict
span_delete_partial_response_from_dict = SpanDeletePartialResponse.from_dict(span_delete_partial_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


