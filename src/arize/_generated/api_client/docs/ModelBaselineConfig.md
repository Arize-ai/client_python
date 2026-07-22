# ModelBaselineConfig

Uses the model's primary baseline as the comparison dataset. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**filters** | [**List[MonitorFilter]**](MonitorFilter.md) | Filters applied to the comparison dataset. An empty array means no comparison dataset filters are configured.  | 

## Example

```python
from arize._generated.api_client.models.model_baseline_config import ModelBaselineConfig

# TODO update the JSON string below
json = "{}"
# create an instance of ModelBaselineConfig from a JSON string
model_baseline_config_instance = ModelBaselineConfig.from_json(json)
# print the JSON string representation of the object
print(ModelBaselineConfig.to_json())

# convert the object into a dict
model_baseline_config_dict = model_baseline_config_instance.to_dict()
# create an instance of ModelBaselineConfig from a dict
model_baseline_config_from_dict = ModelBaselineConfig.from_dict(model_baseline_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


