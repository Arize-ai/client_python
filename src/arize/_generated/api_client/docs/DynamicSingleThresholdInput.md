# DynamicSingleThresholdInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this threshold as a single value derived from the metric&#39;s history. | 
**calculation** | [**ThresholdCalculation**](ThresholdCalculation.md) | The statistical calculation the threshold is derived from. | 
**operator** | [**ThresholdOperator**](ThresholdOperator.md) | Comparison between the computed metric and the derived threshold. Any operator except &#x60;NOT_EQUALS&#x60;. | 
**multiplier** | **float** | Number of deviations from the baseline. Greater than 0 and at most 10. | 

## Example

```python
from arize._generated.api_client.models.dynamic_single_threshold_input import DynamicSingleThresholdInput

# TODO update the JSON string below
json = "{}"
# create an instance of DynamicSingleThresholdInput from a JSON string
dynamic_single_threshold_input_instance = DynamicSingleThresholdInput.from_json(json)
# print the JSON string representation of the object
print(DynamicSingleThresholdInput.to_json())

# convert the object into a dict
dynamic_single_threshold_input_dict = dynamic_single_threshold_input_instance.to_dict()
# create an instance of DynamicSingleThresholdInput from a dict
dynamic_single_threshold_input_from_dict = DynamicSingleThresholdInput.from_dict(dynamic_single_threshold_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


