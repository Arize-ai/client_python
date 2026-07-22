# DynamicSingleThreshold


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**calculation** | [**ThresholdCalculation**](ThresholdCalculation.md) |  | 
**operator** | [**ThresholdOperator**](ThresholdOperator.md) |  | 
**multiplier** | **float** | The multiplier applied to the calculation (e.g. number of standard deviations) to derive the threshold.  | 

## Example

```python
from arize._generated.api_client.models.dynamic_single_threshold import DynamicSingleThreshold

# TODO update the JSON string below
json = "{}"
# create an instance of DynamicSingleThreshold from a JSON string
dynamic_single_threshold_instance = DynamicSingleThreshold.from_json(json)
# print the JSON string representation of the object
print(DynamicSingleThreshold.to_json())

# convert the object into a dict
dynamic_single_threshold_dict = dynamic_single_threshold_instance.to_dict()
# create an instance of DynamicSingleThreshold from a dict
dynamic_single_threshold_from_dict = DynamicSingleThreshold.from_dict(dynamic_single_threshold_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


