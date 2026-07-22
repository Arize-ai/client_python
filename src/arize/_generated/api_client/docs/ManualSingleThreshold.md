# ManualSingleThreshold


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**operator** | [**ThresholdOperator**](ThresholdOperator.md) |  | 
**value** | **float** | The threshold value the computed metric is compared against. | 

## Example

```python
from arize._generated.api_client.models.manual_single_threshold import ManualSingleThreshold

# TODO update the JSON string below
json = "{}"
# create an instance of ManualSingleThreshold from a JSON string
manual_single_threshold_instance = ManualSingleThreshold.from_json(json)
# print the JSON string representation of the object
print(ManualSingleThreshold.to_json())

# convert the object into a dict
manual_single_threshold_dict = manual_single_threshold_instance.to_dict()
# create an instance of ManualSingleThreshold from a dict
manual_single_threshold_from_dict = ManualSingleThreshold.from_dict(manual_single_threshold_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


