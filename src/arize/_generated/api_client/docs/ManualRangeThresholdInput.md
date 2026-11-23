# ManualRangeThresholdInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this threshold as a manually-set lower/upper bound range. | 
**lower** | [**ManualThresholdBoundInput**](ManualThresholdBoundInput.md) | The lower bound. Its operator must be &#x60;GREATER_THAN&#x60; or &#x60;GREATER_THAN_OR_EQUAL&#x60;, and its value must be below &#x60;upper.value&#x60;.  | 
**upper** | [**ManualThresholdBoundInput**](ManualThresholdBoundInput.md) | The upper bound. Its operator must be &#x60;LESS_THAN&#x60; or &#x60;LESS_THAN_OR_EQUAL&#x60;. | 

## Example

```python
from arize._generated.api_client.models.manual_range_threshold_input import ManualRangeThresholdInput

# TODO update the JSON string below
json = "{}"
# create an instance of ManualRangeThresholdInput from a JSON string
manual_range_threshold_input_instance = ManualRangeThresholdInput.from_json(json)
# print the JSON string representation of the object
print(ManualRangeThresholdInput.to_json())

# convert the object into a dict
manual_range_threshold_input_dict = manual_range_threshold_input_instance.to_dict()
# create an instance of ManualRangeThresholdInput from a dict
manual_range_threshold_input_from_dict = ManualRangeThresholdInput.from_dict(manual_range_threshold_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


