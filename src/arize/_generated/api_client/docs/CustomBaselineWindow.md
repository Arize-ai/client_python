# CustomBaselineWindow

The custom comparison window. The `type` field determines whether the window is fixed or moving. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**fixed_start_date** | **datetime** | The start of the fixed comparison window. | 
**fixed_end_date** | **datetime** | The end of the fixed comparison window. | 
**moving_window_seconds** | **int** | The length of the moving comparison window, in seconds. | 
**moving_window_delay_seconds** | **int** | The delay before the moving comparison window, in seconds. | 

## Example

```python
from arize._generated.api_client.models.custom_baseline_window import CustomBaselineWindow

# TODO update the JSON string below
json = "{}"
# create an instance of CustomBaselineWindow from a JSON string
custom_baseline_window_instance = CustomBaselineWindow.from_json(json)
# print the JSON string representation of the object
print(CustomBaselineWindow.to_json())

# convert the object into a dict
custom_baseline_window_dict = custom_baseline_window_instance.to_dict()
# create an instance of CustomBaselineWindow from a dict
custom_baseline_window_from_dict = CustomBaselineWindow.from_dict(custom_baseline_window_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


