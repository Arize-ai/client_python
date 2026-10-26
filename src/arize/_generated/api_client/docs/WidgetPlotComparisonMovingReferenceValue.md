# WidgetPlotComparisonMovingReferenceValue

Moving time range used as the comparison reference.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies a moving comparison reference. | 
**environment_name** | **str** | Model environment containing the reference data. | 
**external_batch_ids** | **List[str]** | Batches included in the reference. | 
**external_model_version_ids** | **List[str]** | Model versions included in the reference. | 
**model_version_environment_metadata_id** | **str** | Model lineage record defining the reference, when applicable. | [optional] 
**moving_window_ms** | **float** | Duration of the moving reference range in milliseconds. | 
**moving_window_delay_ms** | **float** | Delay between the current time and the moving reference range in milliseconds. | 

## Example

```python
from arize._generated.api_client.models.widget_plot_comparison_moving_reference_value import WidgetPlotComparisonMovingReferenceValue

# TODO update the JSON string below
json = "{}"
# create an instance of WidgetPlotComparisonMovingReferenceValue from a JSON string
widget_plot_comparison_moving_reference_value_instance = WidgetPlotComparisonMovingReferenceValue.from_json(json)
# print the JSON string representation of the object
print(WidgetPlotComparisonMovingReferenceValue.to_json())

# convert the object into a dict
widget_plot_comparison_moving_reference_value_dict = widget_plot_comparison_moving_reference_value_instance.to_dict()
# create an instance of WidgetPlotComparisonMovingReferenceValue from a dict
widget_plot_comparison_moving_reference_value_from_dict = WidgetPlotComparisonMovingReferenceValue.from_dict(widget_plot_comparison_moving_reference_value_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


