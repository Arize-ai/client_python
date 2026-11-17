# CreateBarChartWidgetRequest

Request body for creating a bar chart widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Widget type. Must be &#x60;BAR_CHART&#x60;. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Widget position as [rowStart, colStart, rowEnd, colEnd]. The next available position is used when omitted. | [optional] 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Whether the widget is created as a draft or published. Defaults to &#x60;PUBLISHED&#x60;. | [optional] 
**plots** | [**List[BarChartPlotCreateRequest]**](BarChartPlotCreateRequest.md) | Metric series displayed by the chart. | 
**is_normalized** | **bool** | Whether bar values are normalized to percentages. | [optional] 
**performance_metric** | **str** | Performance metric displayed by the widget. | [optional] 
**bin_option** | **str** | Binning strategy applied to the dimension values. | [optional] 
**num_bins** | **int** | Number of bins when the binning strategy requires one. | [optional] 
**custom_bins** | **List[float]** | Custom bin boundaries. | [optional] 
**quantiles** | **List[float]** | Quantile boundaries used when binning by quantile. | [optional] 
**sort_order** | **str** | Order in which bars are sorted. | [optional] 
**top_n** | **int** | Limits the chart to the top N values. | [optional] 
**is_stacked** | **bool** | Whether bars are stacked by the stack dimension. | [optional] 
**y_min** | **float** | Lower bound of the y axis. The axis range is automatic when omitted. | [optional] 
**y_max** | **float** | Upper bound of the y axis. The axis range is automatic when omitted. | [optional] 
**y_axis_label** | **str** | Label displayed on the y axis. | [optional] 
**x_axis_label** | **str** | Label displayed on the x axis. | [optional] 
**x_axis_tilt** | **str** | Rotation applied to the x axis labels. | [optional] 
**font_size** | **str** | Font size used for chart text. | [optional] 
**show_legend** | **bool** | Whether the chart legend is displayed. | [optional] 
**legend_placement** | **str** | Position of the chart legend. | [optional] 
**timezone** | **str** | IANA timezone used to interpret the time range, for example &#x60;America/Los_Angeles&#x60;. | [optional] 
**start_time** | **datetime** | Start of a custom time range. Used with &#x60;end_time&#x60; when the widget overrides the dashboard time range. | [optional] 
**end_time** | **datetime** | End of a custom time range. Used with &#x60;start_time&#x60; when the widget overrides the dashboard time range. | [optional] 
**time_series_data_granularity** | **str** | Granularity of the time series buckets, for example &#x60;hour&#x60; or &#x60;day&#x60;. | [optional] 
**overwrite_global_time** | **bool** | Whether the widget uses its own time range instead of the dashboard time range. | [optional] 
**time_range_key** | **str** | Relative time range key, for example &#x60;7d&#x60;, or &#x60;custom&#x60; to use &#x60;start_time&#x60; and &#x60;end_time&#x60;. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_bar_chart_widget_request import CreateBarChartWidgetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateBarChartWidgetRequest from a JSON string
create_bar_chart_widget_request_instance = CreateBarChartWidgetRequest.from_json(json)
# print the JSON string representation of the object
print(CreateBarChartWidgetRequest.to_json())

# convert the object into a dict
create_bar_chart_widget_request_dict = create_bar_chart_widget_request_instance.to_dict()
# create an instance of CreateBarChartWidgetRequest from a dict
create_bar_chart_widget_request_from_dict = CreateBarChartWidgetRequest.from_dict(create_bar_chart_widget_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


