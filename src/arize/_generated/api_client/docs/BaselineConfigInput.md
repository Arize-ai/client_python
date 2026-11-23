# BaselineConfigInput

The comparison dataset a metric is evaluated against. `type` selects the project's primary baseline or a custom production window. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Compare against the project&#39;s primary baseline. Rejected when the project has no primary baseline.  | 
**window** | [**CustomBaselineWindowInput**](CustomBaselineWindowInput.md) |  | 
**model_versions** | **List[str]** | External model version IDs the data is limited to, at most 100 with no duplicates, each an existing version of the project. Defaults to all versions.  | [optional] 
**filters** | [**List[MlFilterInput]**](MlFilterInput.md) | Filters narrowing the comparison dataset, on fields in the project&#39;s production data. At most 25. Defaults to no filters. | [optional] 

## Example

```python
from arize._generated.api_client.models.baseline_config_input import BaselineConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of BaselineConfigInput from a JSON string
baseline_config_input_instance = BaselineConfigInput.from_json(json)
# print the JSON string representation of the object
print(BaselineConfigInput.to_json())

# convert the object into a dict
baseline_config_input_dict = baseline_config_input_instance.to_dict()
# create an instance of BaselineConfigInput from a dict
baseline_config_input_from_dict = BaselineConfigInput.from_dict(baseline_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


