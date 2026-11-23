# TracingFilterInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dimension** | [**TracingAttributeInput**](TracingAttributeInput.md) | The attribute the filter applies to. | 
**operator** | [**MonitorFilterInputOperator**](MonitorFilterInputOperator.md) | Comparison applied to the attribute. Numeric attributes support &#x60;EQUALS&#x60;, &#x60;NOT_EQUALS&#x60;, &#x60;GREATER_THAN&#x60;, &#x60;GREATER_THAN_OR_EQUAL&#x60;, &#x60;LESS_THAN&#x60; and &#x60;LESS_THAN_OR_EQUAL&#x60;; string attributes support &#x60;EQUALS&#x60;, &#x60;NOT_EQUALS&#x60; and &#x60;CONTAINS_STRING&#x60;; string-list attributes support &#x60;CONTAINS&#x60;.  | 
**values** | **List[str]** | Values compared against, at most 100, each 1 to 1000 characters. Numeric attributes take decimal strings such as &#x60;\&quot;0.5\&quot;&#x60;. The &#x60;GREATER_THAN&#x60;, &#x60;GREATER_THAN_OR_EQUAL&#x60;, &#x60;LESS_THAN&#x60; and &#x60;LESS_THAN_OR_EQUAL&#x60; operators take exactly one value.  | 

## Example

```python
from arize._generated.api_client.models.tracing_filter_input import TracingFilterInput

# TODO update the JSON string below
json = "{}"
# create an instance of TracingFilterInput from a JSON string
tracing_filter_input_instance = TracingFilterInput.from_json(json)
# print the JSON string representation of the object
print(TracingFilterInput.to_json())

# convert the object into a dict
tracing_filter_input_dict = tracing_filter_input_instance.to_dict()
# create an instance of TracingFilterInput from a dict
tracing_filter_input_from_dict = TracingFilterInput.from_dict(tracing_filter_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


