# ManualRangeThreshold


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**lower** | [**ManualThresholdBound**](ManualThresholdBound.md) | The lower bound of the range. Its operator must be &#x60;GREATER_THAN&#x60; or &#x60;GREATER_THAN_OR_EQUAL&#x60;.  | 
**upper** | [**ManualThresholdBound**](ManualThresholdBound.md) | The upper bound of the range. Its &#x60;operator&#x60; must be &#x60;LESS_THAN&#x60; or &#x60;LESS_THAN_OR_EQUAL&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.manual_range_threshold import ManualRangeThreshold

# TODO update the JSON string below
json = "{}"
# create an instance of ManualRangeThreshold from a JSON string
manual_range_threshold_instance = ManualRangeThreshold.from_json(json)
# print the JSON string representation of the object
print(ManualRangeThreshold.to_json())

# convert the object into a dict
manual_range_threshold_dict = manual_range_threshold_instance.to_dict()
# create an instance of ManualRangeThreshold from a dict
manual_range_threshold_from_dict = ManualRangeThreshold.from_dict(manual_range_threshold_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


