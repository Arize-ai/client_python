# DynamicRangeThresholdInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this threshold as a lower/upper bound range derived from the metric&#39;s history. | 
**calculation** | [**ThresholdCalculation**](ThresholdCalculation.md) | The statistical calculation both bounds are derived from. | 
**lower** | [**DynamicThresholdBoundInput**](DynamicThresholdBoundInput.md) | The lower bound. Its operator must be &#x60;GREATER_THAN&#x60; or &#x60;GREATER_THAN_OR_EQUAL&#x60;. | 
**upper** | [**DynamicThresholdBoundInput**](DynamicThresholdBoundInput.md) | The upper bound. Its operator must be &#x60;LESS_THAN&#x60; or &#x60;LESS_THAN_OR_EQUAL&#x60;. | 

## Example

```python
from arize._generated.api_client.models.dynamic_range_threshold_input import DynamicRangeThresholdInput

# TODO update the JSON string below
json = "{}"
# create an instance of DynamicRangeThresholdInput from a JSON string
dynamic_range_threshold_input_instance = DynamicRangeThresholdInput.from_json(json)
# print the JSON string representation of the object
print(DynamicRangeThresholdInput.to_json())

# convert the object into a dict
dynamic_range_threshold_input_dict = dynamic_range_threshold_input_instance.to_dict()
# create an instance of DynamicRangeThresholdInput from a dict
dynamic_range_threshold_input_from_dict = DynamicRangeThresholdInput.from_dict(dynamic_range_threshold_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


