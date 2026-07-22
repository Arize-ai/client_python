# ThresholdConfig

The monitor's threshold. The `type` field discriminates whether the threshold is manual or dynamic, and single or a bounded range. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**operator** | [**ThresholdOperator**](ThresholdOperator.md) |  | 
**value** | **float** | The threshold value the computed metric is compared against. | 
**calculation** | [**ThresholdCalculation**](ThresholdCalculation.md) |  | 
**multiplier** | **float** | The multiplier applied to the calculation (e.g. number of standard deviations) to derive the threshold.  | 
**lower** | [**DynamicThresholdBound**](DynamicThresholdBound.md) | The lower bound of the range. Its &#x60;operator&#x60; must be &#x60;GREATER_THAN&#x60; or &#x60;GREATER_THAN_OR_EQUAL&#x60;.  | 
**upper** | [**DynamicThresholdBound**](DynamicThresholdBound.md) | The upper bound of the range. Its &#x60;operator&#x60; must be &#x60;LESS_THAN&#x60; or &#x60;LESS_THAN_OR_EQUAL&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.threshold_config import ThresholdConfig

# TODO update the JSON string below
json = "{}"
# create an instance of ThresholdConfig from a JSON string
threshold_config_instance = ThresholdConfig.from_json(json)
# print the JSON string representation of the object
print(ThresholdConfig.to_json())

# convert the object into a dict
threshold_config_dict = threshold_config_instance.to_dict()
# create an instance of ThresholdConfig from a dict
threshold_config_from_dict = ThresholdConfig.from_dict(threshold_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


