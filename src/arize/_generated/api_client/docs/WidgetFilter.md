# WidgetFilter

A saved condition that limits the data used by a widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**filter_type** | **str** | Kind of filter and value representation. | 
**dimension_name** | **str** | Dimension to which the filter applies. | [optional] 
**dimension_data_type** | **str** | Data type of the filtered dimension. | [optional] 
**operator** | **str** | Comparison applied by the filter. | [optional] 
**dimension_values** | **List[str]** | General dimension values used by the filter. | [optional] 
**binary_values** | **List[str]** | Binary values used by the filter. | [optional] 
**numeric_values** | **List[float]** | Numeric values used by the filter. | [optional] 
**categorical_values** | **List[str]** | Categorical values used by the filter. | [optional] 

## Example

```python
from arize._generated.api_client.models.widget_filter import WidgetFilter

# TODO update the JSON string below
json = "{}"
# create an instance of WidgetFilter from a JSON string
widget_filter_instance = WidgetFilter.from_json(json)
# print the JSON string representation of the object
print(WidgetFilter.to_json())

# convert the object into a dict
widget_filter_dict = widget_filter_instance.to_dict()
# create an instance of WidgetFilter from a dict
widget_filter_from_dict = WidgetFilter.from_dict(widget_filter_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


