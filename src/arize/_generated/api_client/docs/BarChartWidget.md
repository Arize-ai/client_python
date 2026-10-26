# BarChartWidget

A dashboard widget that compares metric values across categories.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the widget. | 
**type** | [**WidgetType**](WidgetType.md) | Identifies this widget as a bar chart. | 
**dashboard_id** | **str** | Unique identifier for the dashboard containing the widget. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Position and size of the widget in the dashboard grid. | 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Publication state of the widget. | 
**created_at** | **datetime** | Timestamp when the widget was created. | 
**creator_user_id** | **str** | Unique identifier for the user who created the widget. | 
**is_normalized** | **bool** | Whether values are normalized before display. | 
**performance_metric** | **str** | Performance metric displayed by the chart. | [optional] 
**bin_option** | **str** | Saved binning strategy for numeric values. | [optional] 
**num_bins** | **int** | Number of bins used when values are grouped numerically. | [optional] 
**custom_bins** | **List[float]** | Custom numeric bin boundaries. | [optional] 
**quantiles** | **List[float]** | Quantile boundaries used to group values. | [optional] 
**sort_order** | **str** | Saved ordering of chart categories. | [optional] 
**top_n** | **int** | Maximum number of categories displayed. | [optional] 
**is_stacked** | **bool** | Whether series are stacked in each category. | 
**plots** | [**List[WidgetPlot]**](WidgetPlot.md) | Metric series displayed by the widget. | 
**y_min** | **float** | Minimum value displayed on the y-axis, or null for automatic scaling. | [optional] 
**y_max** | **float** | Maximum value displayed on the y-axis, or null for automatic scaling. | [optional] 
**y_axis_label** | **str** | Custom y-axis label, or null to use the default. | [optional] 
**x_axis_label** | **str** | Custom x-axis label, or null to use the default. | [optional] 
**x_axis_tilt** | **str** | Saved rotation setting for x-axis labels. | [optional] 
**font_size** | **str** | Saved chart font-size setting. | [optional] 
**show_legend** | **bool** | Whether the legend is shown, or null to use the default. | [optional] 
**legend_placement** | **str** | Saved placement of the chart legend. | [optional] 
**timezone** | **str** | Timezone used to display time-series values. | [optional] 
**start_time** | **datetime** | Widget-specific range start, or null when the dashboard range applies. | [optional] 
**end_time** | **datetime** | Widget-specific range end, or null when the dashboard range applies. | [optional] 
**time_series_data_granularity** | **str** | Saved aggregation interval for time-series values. | [optional] 
**overwrite_global_time** | **bool** | Whether this widget uses its own time range instead of the dashboard range. | [optional] 
**time_range_key** | **str** | Saved relative time-range selection. | [optional] 

## Example

```python
from arize._generated.api_client.models.bar_chart_widget import BarChartWidget

# TODO update the JSON string below
json = "{}"
# create an instance of BarChartWidget from a JSON string
bar_chart_widget_instance = BarChartWidget.from_json(json)
# print the JSON string representation of the object
print(BarChartWidget.to_json())

# convert the object into a dict
bar_chart_widget_dict = bar_chart_widget_instance.to_dict()
# create an instance of BarChartWidget from a dict
bar_chart_widget_from_dict = BarChartWidget.from_dict(bar_chart_widget_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


