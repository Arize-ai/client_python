# DynamicRangeThreshold


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**calculation** | [**ThresholdCalculation**](ThresholdCalculation.md) |  | 
**lower** | [**DynamicThresholdBound**](DynamicThresholdBound.md) | The lower bound of the range. Its &#x60;operator&#x60; must be &#x60;GREATER_THAN&#x60; or &#x60;GREATER_THAN_OR_EQUAL&#x60;.  | 
**upper** | [**DynamicThresholdBound**](DynamicThresholdBound.md) | The upper bound of the range. Its &#x60;operator&#x60; must be &#x60;LESS_THAN&#x60; or &#x60;LESS_THAN_OR_EQUAL&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.dynamic_range_threshold import DynamicRangeThreshold

# TODO update the JSON string below
json = "{}"
# create an instance of DynamicRangeThreshold from a JSON string
dynamic_range_threshold_instance = DynamicRangeThreshold.from_json(json)
# print the JSON string representation of the object
print(DynamicRangeThreshold.to_json())

# convert the object into a dict
dynamic_range_threshold_dict = dynamic_range_threshold_instance.to_dict()
# create an instance of DynamicRangeThreshold from a dict
dynamic_range_threshold_from_dict = DynamicRangeThreshold.from_dict(dynamic_range_threshold_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


