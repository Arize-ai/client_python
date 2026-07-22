# CustomBaselineConfig

Uses a custom fixed or moving window as the comparison dataset. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**filters** | [**List[MonitorFilter]**](MonitorFilter.md) | Filters applied to the comparison dataset. An empty array means no comparison dataset filters are configured.  | 
**model_versions** | **List[str]** | Model versions included in the comparison dataset. An empty array means all model versions.  | 
**window** | [**CustomBaselineWindow**](CustomBaselineWindow.md) |  | 

## Example

```python
from arize._generated.api_client.models.custom_baseline_config import CustomBaselineConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CustomBaselineConfig from a JSON string
custom_baseline_config_instance = CustomBaselineConfig.from_json(json)
# print the JSON string representation of the object
print(CustomBaselineConfig.to_json())

# convert the object into a dict
custom_baseline_config_dict = custom_baseline_config_instance.to_dict()
# create an instance of CustomBaselineConfig from a dict
custom_baseline_config_from_dict = CustomBaselineConfig.from_dict(custom_baseline_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


