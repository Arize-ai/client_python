# StaticParam


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Parameter name (matches the managed evaluator&#39;s argument name) | 
**type** | **str** | Argument type for static parameters | 
**default_value** | [**StaticParamDefaultValue**](StaticParamDefaultValue.md) |  | 

## Example

```python
from arize._generated.api_client.models.static_param import StaticParam

# TODO update the JSON string below
json = "{}"
# create an instance of StaticParam from a JSON string
static_param_instance = StaticParam.from_json(json)
# print the JSON string representation of the object
print(StaticParam.to_json())

# convert the object into a dict
static_param_dict = static_param_instance.to_dict()
# create an instance of StaticParam from a dict
static_param_from_dict = StaticParam.from_dict(static_param_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


