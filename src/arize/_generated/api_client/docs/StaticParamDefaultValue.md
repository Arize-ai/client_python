# StaticParamDefaultValue

Default value. Must be a string when `type` is STRING or REGEX, and a string array when `type` is STRING_ARRAY. Mismatches are rejected with 400 by the server. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------

## Example

```python
from arize._generated.api_client.models.static_param_default_value import StaticParamDefaultValue

# TODO update the JSON string below
json = "{}"
# create an instance of StaticParamDefaultValue from a JSON string
static_param_default_value_instance = StaticParamDefaultValue.from_json(json)
# print the JSON string representation of the object
print(StaticParamDefaultValue.to_json())

# convert the object into a dict
static_param_default_value_dict = static_param_default_value_instance.to_dict()
# create an instance of StaticParamDefaultValue from a dict
static_param_default_value_from_dict = StaticParamDefaultValue.from_dict(static_param_default_value_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


