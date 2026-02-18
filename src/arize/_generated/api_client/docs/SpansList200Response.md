# SpansList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**spans** | [**List[Span]**](Span.md) | A list of spans | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.spans_list200_response import SpansList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of SpansList200Response from a JSON string
spans_list200_response_instance = SpansList200Response.from_json(json)
# print the JSON string representation of the object
print(SpansList200Response.to_json())

# convert the object into a dict
spans_list200_response_dict = spans_list200_response_instance.to_dict()
# create an instance of SpansList200Response from a dict
spans_list200_response_from_dict = SpansList200Response.from_dict(spans_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


