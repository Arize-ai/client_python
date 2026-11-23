# ThresholdConfigInput

The monitor's threshold, in the same shape a monitor returns. The `type` field selects a manual or dynamic threshold, single or a bounded range. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this threshold as a single, manually-set value. | 
**operator** | [**ThresholdOperator**](ThresholdOperator.md) | Comparison between the computed metric and the derived threshold. Any operator except &#x60;NOT_EQUALS&#x60;. | 
**value** | **float** | The value the computed metric is compared against. | 
**calculation** | [**ThresholdCalculation**](ThresholdCalculation.md) | The statistical calculation both bounds are derived from. | 
**multiplier** | **float** | Number of deviations from the baseline. Greater than 0 and at most 10. | 
**lower** | [**DynamicThresholdBoundInput**](DynamicThresholdBoundInput.md) | The lower bound. Its operator must be &#x60;GREATER_THAN&#x60; or &#x60;GREATER_THAN_OR_EQUAL&#x60;. | 
**upper** | [**DynamicThresholdBoundInput**](DynamicThresholdBoundInput.md) | The upper bound. Its operator must be &#x60;LESS_THAN&#x60; or &#x60;LESS_THAN_OR_EQUAL&#x60;. | 

## Example

```python
from arize._generated.api_client.models.threshold_config_input import ThresholdConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of ThresholdConfigInput from a JSON string
threshold_config_input_instance = ThresholdConfigInput.from_json(json)
# print the JSON string representation of the object
print(ThresholdConfigInput.to_json())

# convert the object into a dict
threshold_config_input_dict = threshold_config_input_instance.to_dict()
# create an instance of ThresholdConfigInput from a dict
threshold_config_input_from_dict = ThresholdConfigInput.from_dict(threshold_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


