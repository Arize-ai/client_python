# ExperimentChartWidgetPlot

One experiment metric series displayed by an experiment chart widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the plot. | 
**dataset_id** | **str** | Dataset whose experiments are represented by the plot. | 
**position** | **int** | Display order of the plot within the widget. | 
**title** | **str** | Display title of the plot. | 
**performance_metric** | **str** | Performance metric represented by the plot. | [optional] 

## Example

```python
from arize._generated.api_client.models.experiment_chart_widget_plot import ExperimentChartWidgetPlot

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentChartWidgetPlot from a JSON string
experiment_chart_widget_plot_instance = ExperimentChartWidgetPlot.from_json(json)
# print the JSON string representation of the object
print(ExperimentChartWidgetPlot.to_json())

# convert the object into a dict
experiment_chart_widget_plot_dict = experiment_chart_widget_plot_instance.to_dict()
# create an instance of ExperimentChartWidgetPlot from a dict
experiment_chart_widget_plot_from_dict = ExperimentChartWidgetPlot.from_dict(experiment_chart_widget_plot_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


