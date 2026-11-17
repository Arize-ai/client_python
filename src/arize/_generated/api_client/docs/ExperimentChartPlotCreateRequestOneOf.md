# ExperimentChartPlotCreateRequestOneOf


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dataset_id** | **str** | ID of the dataset for this series. | 
**evaluation_metric** | **str** | Evaluation metric displayed by the series. | 
**title** | **str** | Display title of the series. | 
**position** | **int** | Zero-based display order within the widget. | [optional] 

## Example

```python
from arize._generated.api_client.models.experiment_chart_plot_create_request_one_of import ExperimentChartPlotCreateRequestOneOf

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentChartPlotCreateRequestOneOf from a JSON string
experiment_chart_plot_create_request_one_of_instance = ExperimentChartPlotCreateRequestOneOf.from_json(json)
# print the JSON string representation of the object
print(ExperimentChartPlotCreateRequestOneOf.to_json())

# convert the object into a dict
experiment_chart_plot_create_request_one_of_dict = experiment_chart_plot_create_request_one_of_instance.to_dict()
# create an instance of ExperimentChartPlotCreateRequestOneOf from a dict
experiment_chart_plot_create_request_one_of_from_dict = ExperimentChartPlotCreateRequestOneOf.from_dict(experiment_chart_plot_create_request_one_of_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


