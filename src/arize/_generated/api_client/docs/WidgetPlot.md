# WidgetPlot

A metric series displayed by a line or bar chart widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the plot. | 
**position** | **int** | Display order of the plot within the widget. | 
**title** | **str** | Display title of the plot. | 
**model_id** | **str** | Model used by the plot, or null when the source is a dataset. | [optional] 
**model_version_ids** | **List[str]** | Model versions included in the plot. | 
**model_environment_name** | **str** | Model environment name used by the plot. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | Model lineage records included in the plot. | 
**dimension_category** | **str** | Category of the selected dimension. | [optional] 
**dimension_name** | **str** | Name of the selected dimension. | [optional] 
**dimension_data_type** | **str** | Data type of the selected dimension. | [optional] 
**prediction_value_class** | **str** | Prediction class selected for class-specific metrics. | [optional] 
**metric_id** | **int** | Saved numeric ID for the selected built-in metric. | [optional] 
**metric** | **str** | Selected built-in metric identifier, such as accuracy or count. | [optional] 
**custom_metric_id** | **str** | Custom metric represented by the plot. | [optional] 
**ranking_at_k** | **int** | Ranking cutoff used by ranking metrics. | [optional] 
**comparison_model_baseline_id** | **str** | Model baseline used for comparison metrics. | [optional] 
**comparison_reference_source** | [**WidgetPlotComparisonReferenceSource**](WidgetPlotComparisonReferenceSource.md) | Whether the comparison uses the model&#39;s primary baseline or a custom baseline. | [optional] 
**comparison_reference_value** | [**WidgetPlotComparisonReferenceValue**](WidgetPlotComparisonReferenceValue.md) |  | [optional] 
**metric_aggregation_type** | **str** | Aggregation applied to the selected metric. | [optional] 
**split_by_enabled** | **bool** | Whether the plot is split into separate series. | [optional] 
**split_by_dimension** | **str** | Dimension used to split the plot into series. | [optional] 
**split_by_dimension_category** | **str** | Category of the dimension used to split the plot. | [optional] 
**split_by_overall_metric_enabled** | **bool** | Whether an overall metric series is included with split series. | [optional] 
**split_by_all_values_live_update_enabled** | **bool** | Whether split series automatically include newly observed values. | [optional] 
**cohorts** | **List[str]** | Cohort identifiers represented by the plot. | [optional] 
**colors** | **List[str]** | Saved colors assigned to plot series. | [optional] 
**stack_dimension_name** | **str** | Dimension used to stack values. | [optional] 
**stack_dimension_category** | **str** | Category of the dimension used to stack values. | [optional] 
**x_axis_dimension_name** | **str** | Dimension displayed on the x-axis. | [optional] 
**x_axis_dimension_category** | **str** | Category of the dimension displayed on the x-axis. | [optional] 
**y_axis_dimension_name** | **str** | Dimension displayed on the y-axis. | [optional] 
**y_axis_dimension_category** | **str** | Category of the dimension displayed on the y-axis. | [optional] 
**color_by_dimension_name** | **str** | Dimension used to assign series colors. | [optional] 
**color_by_dimension_category** | **str** | Category of the dimension used to assign series colors. | [optional] 
**dataset_source** | [**DatasetSource**](DatasetSource.md) | Dataset-backed source configuration for the plot. | [optional] 
**filters** | [**List[WidgetFilter]**](WidgetFilter.md) | Filters applied before computing the plot. | 

## Example

```python
from arize._generated.api_client.models.widget_plot import WidgetPlot

# TODO update the JSON string below
json = "{}"
# create an instance of WidgetPlot from a JSON string
widget_plot_instance = WidgetPlot.from_json(json)
# print the JSON string representation of the object
print(WidgetPlot.to_json())

# convert the object into a dict
widget_plot_dict = widget_plot_instance.to_dict()
# create an instance of WidgetPlot from a dict
widget_plot_from_dict = WidgetPlot.from_dict(widget_plot_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


