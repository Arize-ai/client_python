# PivotTableColumn

A saved column definition in a pivot table widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the column. | 
**position** | **int** | Display order of the column, or null when no order is saved. | 
**dimension** | **str** | Dimension or metric represented by the column. | [optional] 
**dimension_category** | **str** | Category of the selected dimension. | [optional] 
**aggregation** | **str** | Aggregation applied to values in the column. | [optional] 
**title** | **str** | Display title of the column. | [optional] 
**dataset_source** | [**DatasetSource**](DatasetSource.md) | Dataset-backed source configuration for the column. | [optional] 

## Example

```python
from arize._generated.api_client.models.pivot_table_column import PivotTableColumn

# TODO update the JSON string below
json = "{}"
# create an instance of PivotTableColumn from a JSON string
pivot_table_column_instance = PivotTableColumn.from_json(json)
# print the JSON string representation of the object
print(PivotTableColumn.to_json())

# convert the object into a dict
pivot_table_column_dict = pivot_table_column_instance.to_dict()
# create an instance of PivotTableColumn from a dict
pivot_table_column_from_dict = PivotTableColumn.from_dict(pivot_table_column_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


