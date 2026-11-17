# CreateDashboardWidgetRequest

Request to create a dashboard widget with its type-specific configuration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Widget type. Must be &#x60;LINE_CHART&#x60;. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Widget position as [rowStart, colStart, rowEnd, colEnd]. The next available position is used when omitted. | [optional] 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Whether the widget is created as a draft or published. Defaults to &#x60;PUBLISHED&#x60;. | [optional] 
**widget_type** | [**LineChartType**](LineChartType.md) | Visualization mode of the line chart. | [optional] 
**plots** | [**List[ExperimentChartPlotCreateRequest]**](ExperimentChartPlotCreateRequest.md) | Experiment metric series displayed by the chart. | 
**time_series_metric_type** | **str** | Category of the metric displayed, for example &#x60;modelDataMetric&#x60;. | 
**monitor_id** | **str** | ID of the monitor displayed. Used by monitor-type line charts. | [optional] 
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
**is_normalized** | **bool** | Whether bar values are normalized to percentages. | [optional] 
**performance_metric** | **str** | Performance metric displayed by the widget. | [optional] 
**bin_option** | **str** | Binning strategy applied to the dimension values. | [optional] 
**num_bins** | **int** | Number of bins when the binning strategy requires one. | [optional] 
**custom_bins** | **List[float]** | Custom bin boundaries. | [optional] 
**quantiles** | **List[float]** | Quantile boundaries used when binning by quantile. | [optional] 
**sort_order** | **str** | Order in which bars are sorted. | [optional] 
**top_n** | **int** | Limits the chart to the top N values. | [optional] 
**is_stacked** | **bool** | Whether bars are stacked by the stack dimension. | [optional] 
**model_id** | **str** | ID of the model whose data the widget displays. | [optional] 
**model_version_ids** | **List[str]** | IDs of the model versions to include. All versions are included when empty. | [optional] 
**model_environment_name** | **str** | Environment of the model data, for example &#x60;production&#x60;, &#x60;validation&#x60; or &#x60;tracing&#x60;. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | IDs of the model version environment metadata to include. | [optional] 
**aggregation** | **str** | Aggregation applied to the metric, for example &#x60;count&#x60; or &#x60;avg&#x60;. | 
**dimension_category** | **str** | Category of the dimension, for example &#x60;prediction&#x60;, &#x60;actual&#x60; or &#x60;featureLabel&#x60;. | [optional] 
**dimension_name** | **str** | Name of the dimension displayed. | [optional] 
**dimension_data_type** | **str** | Data type of the dimension, for example &#x60;STRING&#x60; or &#x60;NUMERIC&#x60;. | [optional] 
**prediction_value_class** | **str** | Prediction class used for class-specific metrics. | [optional] 
**custom_metric_id** | **str** | ID of the custom metric displayed. | [optional] 
**ranking_at_k** | **int** | Value of k for ranking metrics. | [optional] 
**custom_statistic_text** | **str** | Custom text displayed alongside the statistic. | [optional] 
**filters** | [**List[CreateWidgetFilter]**](CreateWidgetFilter.md) | Conditions that limit the data used by the pivot table. | 
**content** | **str** | Markdown content displayed by the widget. | 
**grouped_dimension** | **str** | Name of the dimension rows are grouped by. | [optional] 
**grouped_dimension_category** | **str** | Category of the grouping dimension. | [optional] 
**grouped_dimension_title** | **str** | Display title of the grouping column. | [optional] 
**query_filter** | **str** | Filter expression applied to the pivot table data. | [optional] 
**columns** | [**List[PivotTableColumnCreateRequest]**](PivotTableColumnCreateRequest.md) | Columns displayed by the pivot table. | 

## Example

```python
from arize._generated.api_client.models.create_dashboard_widget_request import CreateDashboardWidgetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateDashboardWidgetRequest from a JSON string
create_dashboard_widget_request_instance = CreateDashboardWidgetRequest.from_json(json)
# print the JSON string representation of the object
print(CreateDashboardWidgetRequest.to_json())

# convert the object into a dict
create_dashboard_widget_request_dict = create_dashboard_widget_request_instance.to_dict()
# create an instance of CreateDashboardWidgetRequest from a dict
create_dashboard_widget_request_from_dict = CreateDashboardWidgetRequest.from_dict(create_dashboard_widget_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


