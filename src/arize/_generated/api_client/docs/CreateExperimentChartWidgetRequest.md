# CreateExperimentChartWidgetRequest

Request body for creating an experiment chart widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Widget type. Must be &#x60;EXPERIMENT_CHART&#x60;. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Widget position as [rowStart, colStart, rowEnd, colEnd]. The next available position is used when omitted. | [optional] 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Whether the widget is created as a draft or published. Defaults to &#x60;PUBLISHED&#x60;. | [optional] 
**plots** | [**List[ExperimentChartPlotCreateRequest]**](ExperimentChartPlotCreateRequest.md) | Experiment metric series displayed by the chart. | 
**font_size** | **str** | Font size used for chart text. | [optional] 
**show_legend** | **bool** | Whether the chart legend is displayed. | [optional] 
**legend_placement** | **str** | Position of the chart legend. | [optional] 
**x_axis_label** | **str** | Label displayed on the x axis. | [optional] 
**x_axis_tilt** | **str** | Rotation applied to the x axis labels. | [optional] 
**y_min** | **float** | Lower bound of the y axis. The axis range is automatic when omitted. | [optional] 
**y_max** | **float** | Upper bound of the y axis. The axis range is automatic when omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_experiment_chart_widget_request import CreateExperimentChartWidgetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateExperimentChartWidgetRequest from a JSON string
create_experiment_chart_widget_request_instance = CreateExperimentChartWidgetRequest.from_json(json)
# print the JSON string representation of the object
print(CreateExperimentChartWidgetRequest.to_json())

# convert the object into a dict
create_experiment_chart_widget_request_dict = create_experiment_chart_widget_request_instance.to_dict()
# create an instance of CreateExperimentChartWidgetRequest from a dict
create_experiment_chart_widget_request_from_dict = CreateExperimentChartWidgetRequest.from_dict(create_experiment_chart_widget_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


