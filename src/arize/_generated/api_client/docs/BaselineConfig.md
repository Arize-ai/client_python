# BaselineConfig

The comparison dataset configuration used by drift and comparison-based data quality monitors. The `type` field determines whether the comparison dataset uses the model's primary baseline or a custom fixed or moving window.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**filters** | [**List[MonitorFilter]**](MonitorFilter.md) | Filters applied to the comparison dataset. An empty array means no comparison dataset filters are configured.  | 
**model_versions** | **List[str]** | Model versions included in the comparison dataset. An empty array means all model versions.  | 
**window** | [**CustomBaselineWindow**](CustomBaselineWindow.md) |  | 

## Example

```python
from arize._generated.api_client.models.baseline_config import BaselineConfig

# TODO update the JSON string below
json = "{}"
# create an instance of BaselineConfig from a JSON string
baseline_config_instance = BaselineConfig.from_json(json)
# print the JSON string representation of the object
print(BaselineConfig.to_json())

# convert the object into a dict
baseline_config_dict = baseline_config_instance.to_dict()
# create an instance of BaselineConfig from a dict
baseline_config_from_dict = BaselineConfig.from_dict(baseline_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


