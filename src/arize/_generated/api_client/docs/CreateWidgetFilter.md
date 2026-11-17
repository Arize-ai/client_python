# CreateWidgetFilter

A condition that limits the data used by a plot or widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**filter_type** | **str** | Kind of filter, for example &#x60;dimension&#x60; or &#x60;prediction&#x60;. | 
**dimension_name** | **str** | Name of the dimension displayed. | [optional] 
**dimension_data_type** | **str** | Data type of the dimension, for example &#x60;STRING&#x60; or &#x60;NUMERIC&#x60;. | [optional] 
**operator** | **str** | Comparison operator applied by the filter. | 
**dimension_values** | **List[str]** | Dimension values matched by the filter. | [optional] 
**binary_values** | **List[str]** | Binary values matched by the filter. | [optional] 
**numeric_values** | **List[float]** | Numeric values or bounds used by the filter. | [optional] 
**categorical_values** | **List[str]** | Categorical values matched by the filter. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_widget_filter import CreateWidgetFilter

# TODO update the JSON string below
json = "{}"
# create an instance of CreateWidgetFilter from a JSON string
create_widget_filter_instance = CreateWidgetFilter.from_json(json)
# print the JSON string representation of the object
print(CreateWidgetFilter.to_json())

# convert the object into a dict
create_widget_filter_dict = create_widget_filter_instance.to_dict()
# create an instance of CreateWidgetFilter from a dict
create_widget_filter_from_dict = CreateWidgetFilter.from_dict(create_widget_filter_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


