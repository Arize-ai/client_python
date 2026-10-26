# ExperimentChartWidget

A dashboard widget that compares evaluation metrics across experiments.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the widget. | 
**type** | [**WidgetType**](WidgetType.md) | Identifies this widget as an experiment chart. | 
**dashboard_id** | **str** | Unique identifier for the dashboard containing the widget. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Position and size of the widget in the dashboard grid. | 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Publication state of the widget. | 
**created_at** | **datetime** | Timestamp when the widget was created. | 
**creator_user_id** | **str** | Unique identifier for the user who created the widget. | 
**plots** | [**List[ExperimentChartWidgetPlot]**](ExperimentChartWidgetPlot.md) | Experiment metric series displayed by the widget. | 
**font_size** | **str** | Saved chart font-size setting. | [optional] 
**show_legend** | **bool** | Whether the legend is shown, or null to use the default. | [optional] 
**legend_placement** | **str** | Saved placement of the chart legend. | [optional] 
**x_axis_label** | **str** | Custom x-axis label, or null to use the default. | [optional] 
**x_axis_tilt** | **str** | Saved rotation setting for x-axis labels. | [optional] 
**y_min** | **float** | Minimum value displayed on the y-axis, or null for automatic scaling. | [optional] 
**y_max** | **float** | Maximum value displayed on the y-axis, or null for automatic scaling. | [optional] 

## Example

```python
from arize._generated.api_client.models.experiment_chart_widget import ExperimentChartWidget

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentChartWidget from a JSON string
experiment_chart_widget_instance = ExperimentChartWidget.from_json(json)
# print the JSON string representation of the object
print(ExperimentChartWidget.to_json())

# convert the object into a dict
experiment_chart_widget_dict = experiment_chart_widget_instance.to_dict()
# create an instance of ExperimentChartWidget from a dict
experiment_chart_widget_from_dict = ExperimentChartWidget.from_dict(experiment_chart_widget_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


