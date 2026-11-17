# ExperimentChartPlotCreateRequest

Dataset metric series configuration for an experiment chart.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dataset_id** | **str** | ID of the dataset for this series. | 
**evaluation_metric** | **str** | Evaluation metric displayed by the series. | 
**title** | **str** | Display title of the series. | 
**position** | **int** | Zero-based display order within the widget. | [optional] 
**performance_metric** | **str** | Alias for the evaluation metric displayed by the series. | 

## Example

```python
from arize._generated.api_client.models.experiment_chart_plot_create_request import ExperimentChartPlotCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentChartPlotCreateRequest from a JSON string
experiment_chart_plot_create_request_instance = ExperimentChartPlotCreateRequest.from_json(json)
# print the JSON string representation of the object
print(ExperimentChartPlotCreateRequest.to_json())

# convert the object into a dict
experiment_chart_plot_create_request_dict = experiment_chart_plot_create_request_instance.to_dict()
# create an instance of ExperimentChartPlotCreateRequest from a dict
experiment_chart_plot_create_request_from_dict = ExperimentChartPlotCreateRequest.from_dict(experiment_chart_plot_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


