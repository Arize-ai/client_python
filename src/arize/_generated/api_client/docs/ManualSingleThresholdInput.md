# ManualSingleThresholdInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this threshold as a single, manually-set value. | 
**operator** | [**ThresholdOperator**](ThresholdOperator.md) | Comparison between the computed metric and &#x60;value&#x60;. Any operator except &#x60;NOT_EQUALS&#x60;. | 
**value** | **float** | The value the computed metric is compared against. | 

## Example

```python
from arize._generated.api_client.models.manual_single_threshold_input import ManualSingleThresholdInput

# TODO update the JSON string below
json = "{}"
# create an instance of ManualSingleThresholdInput from a JSON string
manual_single_threshold_input_instance = ManualSingleThresholdInput.from_json(json)
# print the JSON string representation of the object
print(ManualSingleThresholdInput.to_json())

# convert the object into a dict
manual_single_threshold_input_dict = manual_single_threshold_input_instance.to_dict()
# create an instance of ManualSingleThresholdInput from a dict
manual_single_threshold_input_from_dict = ManualSingleThresholdInput.from_dict(manual_single_threshold_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


