# SpanContext


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**trace_id** | **str** | Unique identifier for the trace this span belongs to | 
**span_id** | **str** | Unique identifier for the span | 

## Example

```python
from arize._generated.api_client.models.span_context import SpanContext

# TODO update the JSON string below
json = "{}"
# create an instance of SpanContext from a JSON string
span_context_instance = SpanContext.from_json(json)
# print the JSON string representation of the object
print(SpanContext.to_json())

# convert the object into a dict
span_context_dict = span_context_instance.to_dict()
# create an instance of SpanContext from a dict
span_context_from_dict = SpanContext.from_dict(span_context_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


