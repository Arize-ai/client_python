# CreatePivotTableWidgetRequest

Request body for creating a pivot table widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Widget type. Must be &#x60;PIVOT_TABLE&#x60;. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Widget position as [rowStart, colStart, rowEnd, colEnd]. The next available position is used when omitted. | [optional] 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Whether the widget is created as a draft or published. Defaults to &#x60;PUBLISHED&#x60;. | [optional] 
**model_id** | **str** | ID of the model whose data the widget displays. | [optional] 
**model_version_ids** | **List[str]** | IDs of the model versions to include. All versions are included when empty. | [optional] 
**model_environment_name** | **str** | Environment of the model data, for example &#x60;production&#x60;, &#x60;validation&#x60; or &#x60;tracing&#x60;. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | IDs of the model version environment metadata to include. | [optional] 
**grouped_dimension** | **str** | Name of the dimension rows are grouped by. | [optional] 
**grouped_dimension_category** | **str** | Category of the grouping dimension. | [optional] 
**grouped_dimension_title** | **str** | Display title of the grouping column. | [optional] 
**query_filter** | **str** | Filter expression applied to the pivot table data. | [optional] 
**timezone** | **str** | IANA timezone used to interpret the time range, for example &#x60;America/Los_Angeles&#x60;. | [optional] 
**start_time** | **datetime** | Start of a custom time range. Used with &#x60;end_time&#x60; when the widget overrides the dashboard time range. | [optional] 
**end_time** | **datetime** | End of a custom time range. Used with &#x60;start_time&#x60; when the widget overrides the dashboard time range. | [optional] 
**time_range_key** | **str** | Relative time range key, for example &#x60;7d&#x60;, or &#x60;custom&#x60; to use &#x60;start_time&#x60; and &#x60;end_time&#x60;. | [optional] 
**overwrite_global_time** | **bool** | Whether the widget uses its own time range instead of the dashboard time range. | [optional] 
**columns** | [**List[PivotTableColumnCreateRequest]**](PivotTableColumnCreateRequest.md) | Columns displayed by the pivot table. | 
**filters** | [**List[CreateWidgetFilter]**](CreateWidgetFilter.md) | Conditions that limit the data used by the pivot table. | 

## Example

```python
from arize._generated.api_client.models.create_pivot_table_widget_request import CreatePivotTableWidgetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreatePivotTableWidgetRequest from a JSON string
create_pivot_table_widget_request_instance = CreatePivotTableWidgetRequest.from_json(json)
# print the JSON string representation of the object
print(CreatePivotTableWidgetRequest.to_json())

# convert the object into a dict
create_pivot_table_widget_request_dict = create_pivot_table_widget_request_instance.to_dict()
# create an instance of CreatePivotTableWidgetRequest from a dict
create_pivot_table_widget_request_from_dict = CreatePivotTableWidgetRequest.from_dict(create_pivot_table_widget_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


