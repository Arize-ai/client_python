# BarChartPlotCreateRequest

Model or dataset metric series configuration for a bar chart.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**title** | **str** | Display title of the series. | 
**position** | **int** | Zero-based display order within the widget. | [optional] 
**dimension_category** | **str** | Category of the dimension, for example &#x60;prediction&#x60;, &#x60;actual&#x60; or &#x60;featureLabel&#x60;. | 
**model_version_ids** | **List[str]** | IDs of the model versions to include. All versions are included when empty. | [optional] 
**filters** | [**List[CreateWidgetFilter]**](CreateWidgetFilter.md) | Conditions that limit the data used. | [optional] 
**model_id** | **str** | ID of the model whose data the widget displays. | [optional] 
**dimension_name** | **str** | Name of the dimension displayed. | [optional] 
**dimension_data_type** | **str** | Data type of the dimension, for example &#x60;STRING&#x60; or &#x60;NUMERIC&#x60;. | [optional] 
**colors** | **List[str]** | Colors assigned to the series. | [optional] 
**positive_class** | **str** | Class treated as positive for classification metrics. | [optional] 
**ranking_at_k** | **int** | Value of k for ranking metrics. | [optional] 
**stack_dimension_category** | **str** | Category of the dimension bars are stacked by. | [optional] 
**stack_dimension_name** | **str** | Name of the dimension bars are stacked by. | [optional] 
**model_environment_name** | **str** | Environment of the model data, for example &#x60;production&#x60;, &#x60;validation&#x60; or &#x60;tracing&#x60;. | [optional] 
**aggregation** | **str** | Aggregation applied to the metric, for example &#x60;count&#x60; or &#x60;avg&#x60;. | [optional] 
**dataset_source** | [**CreateDatasetSource**](CreateDatasetSource.md) | Dataset fields displayed by the series. Used instead of a model. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | IDs of the model version environment metadata to include. | [optional] 

## Example

```python
from arize._generated.api_client.models.bar_chart_plot_create_request import BarChartPlotCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of BarChartPlotCreateRequest from a JSON string
bar_chart_plot_create_request_instance = BarChartPlotCreateRequest.from_json(json)
# print the JSON string representation of the object
print(BarChartPlotCreateRequest.to_json())

# convert the object into a dict
bar_chart_plot_create_request_dict = bar_chart_plot_create_request_instance.to_dict()
# create an instance of BarChartPlotCreateRequest from a dict
bar_chart_plot_create_request_from_dict = BarChartPlotCreateRequest.from_dict(bar_chart_plot_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


