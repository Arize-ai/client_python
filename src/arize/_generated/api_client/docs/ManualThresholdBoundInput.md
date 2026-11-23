# ManualThresholdBoundInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**operator** | [**ThresholdOperator**](ThresholdOperator.md) | Comparison between the computed metric and this bound. | 
**value** | **float** | The bound value. | 

## Example

```python
from arize._generated.api_client.models.manual_threshold_bound_input import ManualThresholdBoundInput

# TODO update the JSON string below
json = "{}"
# create an instance of ManualThresholdBoundInput from a JSON string
manual_threshold_bound_input_instance = ManualThresholdBoundInput.from_json(json)
# print the JSON string representation of the object
print(ManualThresholdBoundInput.to_json())

# convert the object into a dict
manual_threshold_bound_input_dict = manual_threshold_bound_input_instance.to_dict()
# create an instance of ManualThresholdBoundInput from a dict
manual_threshold_bound_input_from_dict = ManualThresholdBoundInput.from_dict(manual_threshold_bound_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


