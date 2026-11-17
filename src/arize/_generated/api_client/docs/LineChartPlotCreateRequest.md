# LineChartPlotCreateRequest

Model metric series configuration for a line chart.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**model_id** | **str** | ID of the model for this series. | 
**title** | **str** | Display title of the series. | 
**metric** | **str** | Metric displayed by the series. | 
**model_environment_name** | **str** | Environment of the model data, for example &#x60;production&#x60;, &#x60;validation&#x60; or &#x60;tracing&#x60;. | [optional] 
**model_version_ids** | **List[str]** | IDs of the model versions to include. All versions are included when empty. | [optional] 
**position** | **int** | Zero-based display order within the widget. | [optional] 
**filters** | [**List[CreateWidgetFilter]**](CreateWidgetFilter.md) | Conditions that limit the data used. | [optional] 
**dimension_name** | **str** | Name of the dimension displayed. | [optional] 
**dimension_data_type** | **str** | Data type of the dimension, for example &#x60;STRING&#x60; or &#x60;NUMERIC&#x60;. | [optional] 
**dimension_category** | **str** | Category of the dimension, for example &#x60;prediction&#x60;, &#x60;actual&#x60; or &#x60;featureLabel&#x60;. | [optional] 
**colors** | **List[str]** | Colors assigned to the series. | [optional] 
**cohorts** | **List[str]** | Cohorts included in the series. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | IDs of the model version environment metadata to include. | [optional] 
**metric_aggregation_type** | **str** | How the metric is aggregated across values. | [optional] 
**custom_metric_id** | **str** | ID of the custom metric displayed. | [optional] 
**comparison_model_baseline_id** | **str** | ID of the model baseline the series is compared against. | [optional] 
**positive_class** | **str** | Class treated as positive for classification metrics. | [optional] 
**ranking_at_k** | **int** | Value of k for ranking metrics. | [optional] 
**split_by_dimension** | **str** | Name of the dimension the series is split by. | [optional] 
**split_by_dimension_category** | **str** | Category of the split dimension. | [optional] 
**split_by_enabled** | **bool** | Whether the series is split by a dimension. | [optional] 
**split_by_overall_metric_enabled** | **bool** | Whether the overall metric is shown alongside the split series. | [optional] 
**split_by_all_values_live_update_enabled** | **bool** | Whether all split values are updated as new values arrive. | [optional] 
**x_axis_dimension_name** | **str** | Name of the dimension plotted on the x axis. | [optional] 
**x_axis_dimension_category** | **str** | Category of the x axis dimension. | [optional] 
**y_axis_dimension_name** | **str** | Name of the dimension plotted on the y axis. | [optional] 
**y_axis_dimension_category** | **str** | Category of the y axis dimension. | [optional] 
**color_by_dimension_name** | **str** | Name of the dimension used to color points. | [optional] 
**color_by_dimension_category** | **str** | Category of the color dimension. | [optional] 

## Example

```python
from arize._generated.api_client.models.line_chart_plot_create_request import LineChartPlotCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of LineChartPlotCreateRequest from a JSON string
line_chart_plot_create_request_instance = LineChartPlotCreateRequest.from_json(json)
# print the JSON string representation of the object
print(LineChartPlotCreateRequest.to_json())

# convert the object into a dict
line_chart_plot_create_request_dict = line_chart_plot_create_request_instance.to_dict()
# create an instance of LineChartPlotCreateRequest from a dict
line_chart_plot_create_request_from_dict = LineChartPlotCreateRequest.from_dict(line_chart_plot_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


