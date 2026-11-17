# PivotTableColumnCreateRequest

A dimension or metric column displayed by a pivot table.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dimension** | **str** | Name of the dimension the column displays. | 
**aggregation** | **str** | Aggregation applied to the column, for example &#x60;count&#x60;. | 
**position** | **int** | Zero-based display order within the widget. | [optional] 
**title** | **str** | Display title of the column. | [optional] 
**dimension_category** | **str** | Category of the dimension, for example &#x60;prediction&#x60;. | [optional] 

## Example

```python
from arize._generated.api_client.models.pivot_table_column_create_request import PivotTableColumnCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PivotTableColumnCreateRequest from a JSON string
pivot_table_column_create_request_instance = PivotTableColumnCreateRequest.from_json(json)
# print the JSON string representation of the object
print(PivotTableColumnCreateRequest.to_json())

# convert the object into a dict
pivot_table_column_create_request_dict = pivot_table_column_create_request_instance.to_dict()
# create an instance of PivotTableColumnCreateRequest from a dict
pivot_table_column_create_request_from_dict = PivotTableColumnCreateRequest.from_dict(pivot_table_column_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


