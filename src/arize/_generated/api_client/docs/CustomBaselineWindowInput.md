# CustomBaselineWindowInput

The custom comparison window. `type` selects a fixed range or a moving window.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Compare against data between two timestamps. | 
**fixed_start_date** | **datetime** | Start of the comparison range, with a UTC offset. Must be before &#x60;fixed_end_date&#x60;. | 
**fixed_end_date** | **datetime** | End of the comparison range, with a UTC offset. | 
**moving_window_seconds** | **int** | Length of the moving window, in seconds. A whole number of hours from 1 to 19710. | 
**moving_window_delay_seconds** | **int** | How far before the evaluation time the moving window ends, in seconds. A whole number of hours from 0 to 19710. | 

## Example

```python
from arize._generated.api_client.models.custom_baseline_window_input import CustomBaselineWindowInput

# TODO update the JSON string below
json = "{}"
# create an instance of CustomBaselineWindowInput from a JSON string
custom_baseline_window_input_instance = CustomBaselineWindowInput.from_json(json)
# print the JSON string representation of the object
print(CustomBaselineWindowInput.to_json())

# convert the object into a dict
custom_baseline_window_input_dict = custom_baseline_window_input_instance.to_dict()
# create an instance of CustomBaselineWindowInput from a dict
custom_baseline_window_input_from_dict = CustomBaselineWindowInput.from_dict(custom_baseline_window_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


