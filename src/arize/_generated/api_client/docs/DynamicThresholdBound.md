# DynamicThresholdBound


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**operator** | [**ThresholdOperator**](ThresholdOperator.md) |  | 
**multiplier** | **float** | The multiplier applied to the calculation (e.g. number of standard deviations) to derive this bound.  | 

## Example

```python
from arize._generated.api_client.models.dynamic_threshold_bound import DynamicThresholdBound

# TODO update the JSON string below
json = "{}"
# create an instance of DynamicThresholdBound from a JSON string
dynamic_threshold_bound_instance = DynamicThresholdBound.from_json(json)
# print the JSON string representation of the object
print(DynamicThresholdBound.to_json())

# convert the object into a dict
dynamic_threshold_bound_dict = dynamic_threshold_bound_instance.to_dict()
# create an instance of DynamicThresholdBound from a dict
dynamic_threshold_bound_from_dict = DynamicThresholdBound.from_dict(dynamic_threshold_bound_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


