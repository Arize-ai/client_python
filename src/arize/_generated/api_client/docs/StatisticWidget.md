# StatisticWidget

A dashboard widget that displays a single aggregated metric value.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the widget. | 
**type** | [**WidgetType**](WidgetType.md) | Identifies this widget as a statistic. | 
**dashboard_id** | **str** | Unique identifier for the dashboard containing the widget. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Position and size of the widget in the dashboard grid. | 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Publication state of the widget. | 
**created_at** | **datetime** | Timestamp when the widget was created. | 
**creator_user_id** | **str** | Unique identifier for the user who created the widget. | 
**model_id** | **str** | Model used by the statistic, or null when the source is a dataset. | [optional] 
**model_version_ids** | **List[str]** | Model versions included in the calculation. | 
**model_environment_name** | **str** | Model environment name used by the statistic. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | Model lineage records included in the calculation. | 
**aggregation** | **str** | Aggregation applied to the selected metric. | 
**time_series_metric_type** | **str** | Metric used to compute the statistic. | 
**dimension_category** | **str** | Category of the selected dimension. | [optional] 
**dimension_name** | **str** | Name of the selected dimension. | [optional] 
**dimension_data_type** | **str** | Data type of the selected dimension. | [optional] 
**performance_metric** | **str** | Performance metric used by the statistic. | [optional] 
**prediction_value_class** | **str** | Prediction class selected for class-specific metrics. | [optional] 
**custom_metric_id** | **str** | Custom metric used by the statistic. | [optional] 
**ranking_at_k** | **int** | Ranking cutoff used by ranking metrics. | [optional] 
**custom_statistic_text** | **str** | Custom text displayed with the statistic. | [optional] 
**dataset_source** | [**DatasetSource**](DatasetSource.md) | Dataset-backed source configuration for the statistic. | [optional] 
**filters** | [**List[WidgetFilter]**](WidgetFilter.md) | Filters applied before computing the statistic. | 

## Example

```python
from arize._generated.api_client.models.statistic_widget import StatisticWidget

# TODO update the JSON string below
json = "{}"
# create an instance of StatisticWidget from a JSON string
statistic_widget_instance = StatisticWidget.from_json(json)
# print the JSON string representation of the object
print(StatisticWidget.to_json())

# convert the object into a dict
statistic_widget_dict = statistic_widget_instance.to_dict()
# create an instance of StatisticWidget from a dict
statistic_widget_from_dict = StatisticWidget.from_dict(statistic_widget_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


