# CreateStatisticWidgetRequest

Request body for creating a statistic widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Widget type. Must be &#x60;STATISTIC&#x60;. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Widget position as [rowStart, colStart, rowEnd, colEnd]. The next available position is used when omitted. | [optional] 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Whether the widget is created as a draft or published. Defaults to &#x60;PUBLISHED&#x60;. | [optional] 
**model_id** | **str** | ID of the model whose data the widget displays. | [optional] 
**model_version_ids** | **List[str]** | IDs of the model versions to include. All versions are included when empty. | [optional] 
**model_environment_name** | **str** | Environment of the model data, for example &#x60;production&#x60;, &#x60;validation&#x60; or &#x60;tracing&#x60;. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | IDs of the model version environment metadata to include. | [optional] 
**aggregation** | **str** | Aggregation applied to the metric, for example &#x60;count&#x60; or &#x60;avg&#x60;. | 
**time_series_metric_type** | **str** | Category of the metric displayed, for example &#x60;modelDataMetric&#x60;. | 
**dimension_category** | **str** | Category of the dimension, for example &#x60;prediction&#x60;, &#x60;actual&#x60; or &#x60;featureLabel&#x60;. | [optional] 
**dimension_name** | **str** | Name of the dimension displayed. | [optional] 
**dimension_data_type** | **str** | Data type of the dimension, for example &#x60;STRING&#x60; or &#x60;NUMERIC&#x60;. | [optional] 
**performance_metric** | **str** | Performance metric displayed by the widget. | [optional] 
**prediction_value_class** | **str** | Prediction class used for class-specific metrics. | [optional] 
**custom_metric_id** | **str** | ID of the custom metric displayed. | [optional] 
**ranking_at_k** | **int** | Value of k for ranking metrics. | [optional] 
**custom_statistic_text** | **str** | Custom text displayed alongside the statistic. | [optional] 
**filters** | [**List[CreateWidgetFilter]**](CreateWidgetFilter.md) | Conditions that limit the data used by the statistic. | 

## Example

```python
from arize._generated.api_client.models.create_statistic_widget_request import CreateStatisticWidgetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateStatisticWidgetRequest from a JSON string
create_statistic_widget_request_instance = CreateStatisticWidgetRequest.from_json(json)
# print the JSON string representation of the object
print(CreateStatisticWidgetRequest.to_json())

# convert the object into a dict
create_statistic_widget_request_dict = create_statistic_widget_request_instance.to_dict()
# create an instance of CreateStatisticWidgetRequest from a dict
create_statistic_widget_request_from_dict = CreateStatisticWidgetRequest.from_dict(create_statistic_widget_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


