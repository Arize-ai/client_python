# FixedCustomBaselineWindowInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Compare against data between two timestamps. | 
**fixed_start_date** | **datetime** | Start of the comparison range, with a UTC offset. Must be before &#x60;fixed_end_date&#x60;. | 
**fixed_end_date** | **datetime** | End of the comparison range, with a UTC offset. | 

## Example

```python
from arize._generated.api_client.models.fixed_custom_baseline_window_input import FixedCustomBaselineWindowInput

# TODO update the JSON string below
json = "{}"
# create an instance of FixedCustomBaselineWindowInput from a JSON string
fixed_custom_baseline_window_input_instance = FixedCustomBaselineWindowInput.from_json(json)
# print the JSON string representation of the object
print(FixedCustomBaselineWindowInput.to_json())

# convert the object into a dict
fixed_custom_baseline_window_input_dict = fixed_custom_baseline_window_input_instance.to_dict()
# create an instance of FixedCustomBaselineWindowInput from a dict
fixed_custom_baseline_window_input_from_dict = FixedCustomBaselineWindowInput.from_dict(fixed_custom_baseline_window_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


