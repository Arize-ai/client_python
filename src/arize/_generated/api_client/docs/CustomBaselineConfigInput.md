# CustomBaselineConfigInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Compare against a fixed or moving window of production data. | 
**window** | [**CustomBaselineWindowInput**](CustomBaselineWindowInput.md) |  | 
**model_versions** | **List[str]** | External model version IDs the data is limited to, at most 100 with no duplicates, each an existing version of the project. Defaults to all versions.  | [optional] 
**filters** | [**List[MlFilterInput]**](MlFilterInput.md) | Filters narrowing the comparison dataset, on fields in the project&#39;s production data. At most 25. Defaults to no filters. | [optional] 

## Example

```python
from arize._generated.api_client.models.custom_baseline_config_input import CustomBaselineConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of CustomBaselineConfigInput from a JSON string
custom_baseline_config_input_instance = CustomBaselineConfigInput.from_json(json)
# print the JSON string representation of the object
print(CustomBaselineConfigInput.to_json())

# convert the object into a dict
custom_baseline_config_input_dict = custom_baseline_config_input_instance.to_dict()
# create an instance of CustomBaselineConfigInput from a dict
custom_baseline_config_input_from_dict = CustomBaselineConfigInput.from_dict(custom_baseline_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


