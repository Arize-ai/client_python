# PivotTableWidget

A dashboard widget that groups metrics and dimensions into a table.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the widget. | 
**type** | [**WidgetType**](WidgetType.md) | Identifies this widget as a pivot table. | 
**dashboard_id** | **str** | Unique identifier for the dashboard containing the widget. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Position and size of the widget in the dashboard grid. | 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Publication state of the widget. | 
**created_at** | **datetime** | Timestamp when the widget was created. | 
**creator_user_id** | **str** | Unique identifier for the user who created the widget. | 
**model_id** | **str** | Model used by the table, or null when the source is a dataset. | [optional] 
**model_version_ids** | **List[str]** | Model versions included in the table. | 
**model_environment_name** | **str** | Model environment name used by the table. | [optional] 
**model_version_environment_metadata_ids** | **List[str]** | Model lineage records included in the table. | 
**grouped_dimension** | **str** | Dimension used to group table rows. | [optional] 
**grouped_dimension_category** | **str** | Category of the dimension used to group table rows. | [optional] 
**grouped_dimension_title** | **str** | Display title for the grouped dimension. | [optional] 
**query_filter** | **str** | Saved query expression applied to the table. | [optional] 
**timezone** | **str** | Timezone used to display time-based values. | [optional] 
**start_time** | **datetime** | Widget-specific range start, or null when the dashboard range applies. | [optional] 
**end_time** | **datetime** | Widget-specific range end, or null when the dashboard range applies. | [optional] 
**time_range_key** | **str** | Saved relative time-range selection. | [optional] 
**overwrite_global_time** | **bool** | Whether this widget uses its own time range instead of the dashboard range. | [optional] 
**columns** | [**List[PivotTableColumn]**](PivotTableColumn.md) | Ordered columns displayed in the table. | 
**filters** | [**List[WidgetFilter]**](WidgetFilter.md) | Filters applied to table values. | 

## Example

```python
from arize._generated.api_client.models.pivot_table_widget import PivotTableWidget

# TODO update the JSON string below
json = "{}"
# create an instance of PivotTableWidget from a JSON string
pivot_table_widget_instance = PivotTableWidget.from_json(json)
# print the JSON string representation of the object
print(PivotTableWidget.to_json())

# convert the object into a dict
pivot_table_widget_dict = pivot_table_widget_instance.to_dict()
# create an instance of PivotTableWidget from a dict
pivot_table_widget_from_dict = PivotTableWidget.from_dict(pivot_table_widget_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


