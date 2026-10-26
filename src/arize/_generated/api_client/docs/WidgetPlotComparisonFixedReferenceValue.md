# WidgetPlotComparisonFixedReferenceValue

Fixed time range used as the comparison reference.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies a fixed comparison reference. | 
**environment_name** | **str** | Model environment containing the reference data. | 
**external_batch_ids** | **List[str]** | Batches included in the reference. | 
**external_model_version_ids** | **List[str]** | Model versions included in the reference. | 
**model_version_environment_metadata_id** | **str** | Model lineage record defining the reference, when applicable. | [optional] 
**start_time** | **datetime** | Inclusive start of the fixed reference range. | 
**end_time** | **datetime** | Exclusive end of the fixed reference range. | 

## Example

```python
from arize._generated.api_client.models.widget_plot_comparison_fixed_reference_value import WidgetPlotComparisonFixedReferenceValue

# TODO update the JSON string below
json = "{}"
# create an instance of WidgetPlotComparisonFixedReferenceValue from a JSON string
widget_plot_comparison_fixed_reference_value_instance = WidgetPlotComparisonFixedReferenceValue.from_json(json)
# print the JSON string representation of the object
print(WidgetPlotComparisonFixedReferenceValue.to_json())

# convert the object into a dict
widget_plot_comparison_fixed_reference_value_dict = widget_plot_comparison_fixed_reference_value_instance.to_dict()
# create an instance of WidgetPlotComparisonFixedReferenceValue from a dict
widget_plot_comparison_fixed_reference_value_from_dict = WidgetPlotComparisonFixedReferenceValue.from_dict(widget_plot_comparison_fixed_reference_value_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


