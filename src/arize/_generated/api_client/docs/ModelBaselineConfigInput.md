# ModelBaselineConfigInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Compare against the project&#39;s primary baseline. Rejected when the project has no primary baseline.  | 

## Example

```python
from arize._generated.api_client.models.model_baseline_config_input import ModelBaselineConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of ModelBaselineConfigInput from a JSON string
model_baseline_config_input_instance = ModelBaselineConfigInput.from_json(json)
# print the JSON string representation of the object
print(ModelBaselineConfigInput.to_json())

# convert the object into a dict
model_baseline_config_input_dict = model_baseline_config_input_instance.to_dict()
# create an instance of ModelBaselineConfigInput from a dict
model_baseline_config_input_from_dict = ModelBaselineConfigInput.from_dict(model_baseline_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


