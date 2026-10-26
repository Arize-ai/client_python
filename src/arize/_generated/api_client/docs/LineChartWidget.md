# LineChartWidget

A dashboard widget that displays one or more metric series.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the widget. | 
**type** | [**WidgetType**](WidgetType.md) | Identifies this widget as a line chart. | 
**dashboard_id** | **str** | Unique identifier for the dashboard containing the widget. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Position and size of the widget in the dashboard grid. | 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Publication state of the widget. | 
**created_at** | **datetime** | Timestamp when the widget was created. | 
**creator_user_id** | **str** | Unique identifier for the user who created the widget. | 
**widget_type** | [**LineChartType**](LineChartType.md) | Visualization mode used by the chart. | 
**plots** | [**List[WidgetPlot]**](WidgetPlot.md) | Metric series displayed by the widget. | 
**time_series_metric_type** | **str** | Metric used to compute the time series. | 
**monitor_id** | **str** | Monitor represented by the widget when using the monitor chart mode. | [optional] 
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
**time_series_data_granularity** | **str** | Saved aggregation interval for the time series. | [optional] 
**overwrite_global_time** | **bool** | Whether this widget uses its own time range instead of the dashboard range. | [optional] 
**time_range_key** | **str** | Saved relative time-range selection. | [optional] 

## Example

```python
from arize._generated.api_client.models.line_chart_widget import LineChartWidget

# TODO update the JSON string below
json = "{}"
# create an instance of LineChartWidget from a JSON string
line_chart_widget_instance = LineChartWidget.from_json(json)
# print the JSON string representation of the object
print(LineChartWidget.to_json())

# convert the object into a dict
line_chart_widget_dict = line_chart_widget_instance.to_dict()
# create an instance of LineChartWidget from a dict
line_chart_widget_from_dict = LineChartWidget.from_dict(line_chart_widget_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


