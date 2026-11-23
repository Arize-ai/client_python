# MlFilterInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dimension** | [**MlDimensionInput**](MlDimensionInput.md) | The field the filter applies to. Embedding features cannot be filtered. | 
**operator** | [**MonitorFilterInputOperator**](MonitorFilterInputOperator.md) | Comparison applied to the field. Numeric fields support &#x60;EQUALS&#x60;, &#x60;NOT_EQUALS&#x60;, &#x60;GREATER_THAN&#x60;, &#x60;GREATER_THAN_OR_EQUAL&#x60;, &#x60;LESS_THAN&#x60; and &#x60;LESS_THAN_OR_EQUAL&#x60;; categorical fields support &#x60;EQUALS&#x60;, &#x60;NOT_EQUALS&#x60; and &#x60;CONTAINS_STRING&#x60;; string-list fields support &#x60;CONTAINS&#x60;.  | 
**values** | **List[str]** | Values compared against, at most 100, each 1 to 1000 characters. Numeric fields take decimal strings such as &#x60;\&quot;0.5\&quot;&#x60;. The &#x60;GREATER_THAN&#x60;, &#x60;GREATER_THAN_OR_EQUAL&#x60;, &#x60;LESS_THAN&#x60; and &#x60;LESS_THAN_OR_EQUAL&#x60; operators take exactly one value.  | 

## Example

```python
from arize._generated.api_client.models.ml_filter_input import MlFilterInput

# TODO update the JSON string below
json = "{}"
# create an instance of MlFilterInput from a JSON string
ml_filter_input_instance = MlFilterInput.from_json(json)
# print the JSON string representation of the object
print(MlFilterInput.to_json())

# convert the object into a dict
ml_filter_input_dict = ml_filter_input_instance.to_dict()
# create an instance of MlFilterInput from a dict
ml_filter_input_from_dict = MlFilterInput.from_dict(ml_filter_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


