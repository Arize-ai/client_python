# MovingCustomBaselineWindowInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Compare against a window that moves with the evaluation time. | 
**moving_window_seconds** | **int** | Length of the moving window, in seconds. A whole number of hours from 1 to 19710. | 
**moving_window_delay_seconds** | **int** | How far before the evaluation time the moving window ends, in seconds. A whole number of hours from 0 to 19710. | 

## Example

```python
from arize._generated.api_client.models.moving_custom_baseline_window_input import MovingCustomBaselineWindowInput

# TODO update the JSON string below
json = "{}"
# create an instance of MovingCustomBaselineWindowInput from a JSON string
moving_custom_baseline_window_input_instance = MovingCustomBaselineWindowInput.from_json(json)
# print the JSON string representation of the object
print(MovingCustomBaselineWindowInput.to_json())

# convert the object into a dict
moving_custom_baseline_window_input_dict = moving_custom_baseline_window_input_instance.to_dict()
# create an instance of MovingCustomBaselineWindowInput from a dict
moving_custom_baseline_window_input_from_dict = MovingCustomBaselineWindowInput.from_dict(moving_custom_baseline_window_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


