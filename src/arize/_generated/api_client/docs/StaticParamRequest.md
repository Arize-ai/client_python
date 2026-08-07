# StaticParamRequest

Static evaluator parameter in a write request (strict form of StaticParam)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Parameter name (matches the managed evaluator&#39;s argument name) | 
**type** | [**StaticParamType**](StaticParamType.md) |  | 
**default_value** | [**StaticParamDefaultValue**](StaticParamDefaultValue.md) |  | 

## Example

```python
from arize._generated.api_client.models.static_param_request import StaticParamRequest

# TODO update the JSON string below
json = "{}"
# create an instance of StaticParamRequest from a JSON string
static_param_request_instance = StaticParamRequest.from_json(json)
# print the JSON string representation of the object
print(StaticParamRequest.to_json())

# convert the object into a dict
static_param_request_dict = static_param_request_instance.to_dict()
# create an instance of StaticParamRequest from a dict
static_param_request_from_dict = StaticParamRequest.from_dict(static_param_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


