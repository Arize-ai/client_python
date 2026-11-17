# ExperimentChartPlotCreateRequestOneOf1


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dataset_id** | **str** | ID of the dataset for this series. | 
**performance_metric** | **str** | Alias for the evaluation metric displayed by the series. | 
**title** | **str** | Display title of the series. | 
**position** | **int** | Zero-based display order within the widget. | [optional] 

## Example

```python
from arize._generated.api_client.models.experiment_chart_plot_create_request_one_of1 import ExperimentChartPlotCreateRequestOneOf1

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentChartPlotCreateRequestOneOf1 from a JSON string
experiment_chart_plot_create_request_one_of1_instance = ExperimentChartPlotCreateRequestOneOf1.from_json(json)
# print the JSON string representation of the object
print(ExperimentChartPlotCreateRequestOneOf1.to_json())

# convert the object into a dict
experiment_chart_plot_create_request_one_of1_dict = experiment_chart_plot_create_request_one_of1_instance.to_dict()
# create an instance of ExperimentChartPlotCreateRequestOneOf1 from a dict
experiment_chart_plot_create_request_one_of1_from_dict = ExperimentChartPlotCreateRequestOneOf1.from_dict(experiment_chart_plot_create_request_one_of1_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


