# ListSpansResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**spans** | [**List[Span]**](Span.md) | A list of spans | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_spans_response import ListSpansResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListSpansResponse from a JSON string
list_spans_response_instance = ListSpansResponse.from_json(json)
# print the JSON string representation of the object
print(ListSpansResponse.to_json())

# convert the object into a dict
list_spans_response_dict = list_spans_response_instance.to_dict()
# create an instance of ListSpansResponse from a dict
list_spans_response_from_dict = ListSpansResponse.from_dict(list_spans_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


