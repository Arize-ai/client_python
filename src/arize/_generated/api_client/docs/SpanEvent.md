# SpanEvent


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the event | 
**timestamp** | **datetime** | Timestamp when the event occurred | 
**attributes** | **Dict[str, object]** | Key-value pairs of event attributes | [optional] 

## Example

```python
from arize._generated.api_client.models.span_event import SpanEvent

# TODO update the JSON string below
json = "{}"
# create an instance of SpanEvent from a JSON string
span_event_instance = SpanEvent.from_json(json)
# print the JSON string representation of the object
print(SpanEvent.to_json())

# convert the object into a dict
span_event_dict = span_event_instance.to_dict()
# create an instance of SpanEvent from a dict
span_event_from_dict = SpanEvent.from_dict(span_event_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


