# DynamicThresholdBoundInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**operator** | [**ThresholdOperator**](ThresholdOperator.md) | Comparison between the computed metric and this bound. | 
**multiplier** | **float** | Number of deviations from the baseline for this bound. Greater than 0 and at most 10. | 

## Example

```python
from arize._generated.api_client.models.dynamic_threshold_bound_input import DynamicThresholdBoundInput

# TODO update the JSON string below
json = "{}"
# create an instance of DynamicThresholdBoundInput from a JSON string
dynamic_threshold_bound_input_instance = DynamicThresholdBoundInput.from_json(json)
# print the JSON string representation of the object
print(DynamicThresholdBoundInput.to_json())

# convert the object into a dict
dynamic_threshold_bound_input_dict = dynamic_threshold_bound_input_instance.to_dict()
# create an instance of DynamicThresholdBoundInput from a dict
dynamic_threshold_bound_input_from_dict = DynamicThresholdBoundInput.from_dict(dynamic_threshold_bound_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


