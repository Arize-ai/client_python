# Span

A Span represents a single unit of work within a distributed trace for an LLM application. It captures the operation’s input and output, start and end times, and a span kind indicating its role (such as LLM, Tool, Agent, Retriever, Chain, or Embedding). Spans are hierarchically related and combine to form a trace, enabling end-to-end visibility into request execution, performance bottlenecks, and errors across complex LLM pipelines. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the span | 
**context** | [**SpanContext**](SpanContext.md) |  | 
**kind** | **str** | The kind of span (e.g., TOOL, CHAIN, LLM, RETRIEVER, EMBEDDING) | 
**parent_id** | **str** | ID of the parent span | [optional] 
**start_time** | **datetime** | Timestamp when the span started | 
**end_time** | **datetime** | Timestamp when the span ended | 
**status_code** | **str** | Status code of the span | [optional] 
**status_message** | **str** | Status message associated with the span | [optional] 
**attributes** | **Dict[str, object]** | Key-value pairs of span attributes | [optional] 
**annotations** | **Dict[str, object]** | Key-value pairs of span annotations | [optional] 
**evaluations** | **Dict[str, object]** | Key-value pairs of span evaluations | [optional] 
**events** | [**List[SpanEvent]**](SpanEvent.md) | List of events that occurred during the span | [optional] 

## Example

```python
from arize._generated.api_client.models.span import Span

# TODO update the JSON string below
json = "{}"
# create an instance of Span from a JSON string
span_instance = Span.from_json(json)
# print the JSON string representation of the object
print(Span.to_json())

# convert the object into a dict
span_dict = span_instance.to_dict()
# create an instance of Span from a dict
span_from_dict = Span.from_dict(span_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


