# MovingCustomBaselineWindow

A custom comparison dataset using a moving window defined in seconds.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**moving_window_seconds** | **int** | The length of the moving comparison window, in seconds. | 
**moving_window_delay_seconds** | **int** | The delay before the moving comparison window, in seconds. | 

## Example

```python
from arize._generated.api_client.models.moving_custom_baseline_window import MovingCustomBaselineWindow

# TODO update the JSON string below
json = "{}"
# create an instance of MovingCustomBaselineWindow from a JSON string
moving_custom_baseline_window_instance = MovingCustomBaselineWindow.from_json(json)
# print the JSON string representation of the object
print(MovingCustomBaselineWindow.to_json())

# convert the object into a dict
moving_custom_baseline_window_dict = moving_custom_baseline_window_instance.to_dict()
# create an instance of MovingCustomBaselineWindow from a dict
moving_custom_baseline_window_from_dict = MovingCustomBaselineWindow.from_dict(moving_custom_baseline_window_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


