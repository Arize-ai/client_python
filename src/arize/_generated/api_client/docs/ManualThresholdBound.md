# ManualThresholdBound


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**operator** | [**ThresholdOperator**](ThresholdOperator.md) |  | 
**value** | **float** | The bound value the computed metric is compared against. | 

## Example

```python
from arize._generated.api_client.models.manual_threshold_bound import ManualThresholdBound

# TODO update the JSON string below
json = "{}"
# create an instance of ManualThresholdBound from a JSON string
manual_threshold_bound_instance = ManualThresholdBound.from_json(json)
# print the JSON string representation of the object
print(ManualThresholdBound.to_json())

# convert the object into a dict
manual_threshold_bound_dict = manual_threshold_bound_instance.to_dict()
# create an instance of ManualThresholdBound from a dict
manual_threshold_bound_from_dict = ManualThresholdBound.from_dict(manual_threshold_bound_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


