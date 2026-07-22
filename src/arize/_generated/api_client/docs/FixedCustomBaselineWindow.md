# FixedCustomBaselineWindow

A custom comparison dataset using data between a fixed start and end date.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**fixed_start_date** | **datetime** | The start of the fixed comparison window. | 
**fixed_end_date** | **datetime** | The end of the fixed comparison window. | 

## Example

```python
from arize._generated.api_client.models.fixed_custom_baseline_window import FixedCustomBaselineWindow

# TODO update the JSON string below
json = "{}"
# create an instance of FixedCustomBaselineWindow from a JSON string
fixed_custom_baseline_window_instance = FixedCustomBaselineWindow.from_json(json)
# print the JSON string representation of the object
print(FixedCustomBaselineWindow.to_json())

# convert the object into a dict
fixed_custom_baseline_window_dict = fixed_custom_baseline_window_instance.to_dict()
# create an instance of FixedCustomBaselineWindow from a dict
fixed_custom_baseline_window_from_dict = FixedCustomBaselineWindow.from_dict(fixed_custom_baseline_window_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


