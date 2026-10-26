# Widget

A dashboard widget and its type-specific saved configuration.

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
**plots** | [**List[ExperimentChartWidgetPlot]**](ExperimentChartWidgetPlot.md) | Experiment metric series displayed by the widget. | 
**time_series_metric_type** | **str** | Metric used to compute the statistic. | 
**monitor_id** | **str** | Monitor represented by the widget when using the monitor chart mode. | [optional] 
**y_min** | **float** | Minimum value displayed on the y-axis, or null for automatic scaling. | [optional] 
**y_max** | **float** | Maximum value displayed on the y-axis, or null for automatic scaling. | [optional] 
**y_axis_label** | **str** | Custom y-axis label, or null to use the default. | [optional] 
**x_axis_label** | **str** | Custom x-axis label, or null to use the default. | [optional] 
**x_axis_tilt** | **str** | Saved rotation setting for x-axis labels. | [optional] 
**font_size** | **str** | Saved chart font-size setting. | [optional] 
**show_legend** | **bool** | Whether the legend is shown, or null to use the default. | [optional] 
**legend_placement** | **str** | Saved placement of the chart legend. | [optional] 
**timezone** | **str** | Timezone used to display time-based values. | [optional] 
**start_time** | **datetime** | Widget-specific range start, or null when the dashboard range applies. | [optional] 
**end_time** | **datetime** | Widget-specific range end, or null when the dashboard range applies. | [optional] 
**time_series_data_granularity** | **str** | Saved aggregation interval for time-series values. | [optional] 
**overwrite_global_time** | **bool** | Whether this widget uses its own time range instead of the dashboard range. | [optional] 
**time_range_key** | **str** | Saved relative time-range selection. | [optional] 
**is_normalized** | **bool** | Whether values are normalized before display. | 
**performance_metric** | **str** | Performance metric used by the statistic. | [optional] 
**bin_option** | **str** | Saved binning strategy for numeric values. | [optional] 
**num_bins** | **int** | Number of bins used when values are grouped numerically. | [optional] 
**custom_bins** | **List[float]** | Custom numeric bin boundaries. | [optional] 
**quantiles** | **List[float]** | Quantile boundaries used to group values. | [optional] 
**sort_order** | **str** | Saved ordering of chart categories. | [optional] 
**top_n** | **int** | Maximum number of categories displayed. | [optional] 
**is_stacked** | **bool** | Whether series are stacked in each category. | 
**model_id** | **str** | Model used by the table, or null when the source is a dataset. | [optional] 
**model_version_ids** | **List[str]** | Model versions included in the table. | 
**model_environment_name** | **str** | Model environment name used by the table. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | Model lineage records included in the table. | 
**aggregation** | **str** | Aggregation applied to the selected metric. | 
**dimension_category** | **str** | Category of the selected dimension. | [optional] 
**dimension_name** | **str** | Name of the selected dimension. | [optional] 
**dimension_data_type** | **str** | Data type of the selected dimension. | [optional] 
**prediction_value_class** | **str** | Prediction class selected for class-specific metrics. | [optional] 
**custom_metric_id** | **str** | Custom metric used by the statistic. | [optional] 
**ranking_at_k** | **int** | Ranking cutoff used by ranking metrics. | [optional] 
**custom_statistic_text** | **str** | Custom text displayed with the statistic. | [optional] 
**dataset_source** | [**DatasetSource**](DatasetSource.md) | Dataset-backed source configuration for the statistic. | [optional] 
**filters** | [**List[WidgetFilter]**](WidgetFilter.md) | Filters applied to table values. | 
**content** | **str** | Saved text content displayed by the widget. | 
**grouped_dimension** | **str** | Dimension used to group table rows. | [optional] 
**grouped_dimension_category** | **str** | Category of the dimension used to group table rows. | [optional] 
**grouped_dimension_title** | **str** | Display title for the grouped dimension. | [optional] 
**query_filter** | **str** | Saved query expression applied to the table. | [optional] 
**columns** | [**List[PivotTableColumn]**](PivotTableColumn.md) | Ordered columns displayed in the table. | 

## Example

```python
from arize._generated.api_client.models.widget import Widget

# TODO update the JSON string below
json = "{}"
# create an instance of Widget from a JSON string
widget_instance = Widget.from_json(json)
# print the JSON string representation of the object
print(Widget.to_json())

# convert the object into a dict
widget_dict = widget_instance.to_dict()
# create an instance of Widget from a dict
widget_from_dict = Widget.from_dict(widget_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


