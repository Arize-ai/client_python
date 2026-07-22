# MonitorFilter


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dimension** | [**Dimension**](Dimension.md) |  | 
**operator** | [**FilterOperator**](FilterOperator.md) |  | 
**values** | **List[str]** | The values compared against by the operator. | 

## Example

```python
from arize._generated.api_client.models.monitor_filter import MonitorFilter

# TODO update the JSON string below
json = "{}"
# create an instance of MonitorFilter from a JSON string
monitor_filter_instance = MonitorFilter.from_json(json)
# print the JSON string representation of the object
print(MonitorFilter.to_json())

# convert the object into a dict
monitor_filter_dict = monitor_filter_instance.to_dict()
# create an instance of MonitorFilter from a dict
monitor_filter_from_dict = MonitorFilter.from_dict(monitor_filter_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


