# CustomCodeConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_granularity** | **str** | Data granularity level for evaluation. When omitted or null, no granularity filter is applied (span-level evaluation is used by default on the server).  | [optional] 
**query_filter** | **str** | Optional filter query over the chosen data granularity. When omitted or null, no filter is applied.  | [optional] 
**type** | **str** | Discriminator for custom Python code evaluators | 
**name** | **str** | Eval column name. Must match ^[a-zA-Z0-9_\\s\\-&amp;()]+$ | 
**code** | **str** | Python source defining the evaluator class | 
**imports** | **str** | Optional package import block prepended when running the evaluator | [optional] 
**variables** | **List[str]** | Dataset columns or span attributes mapped to evaluate() arguments | 
**static_params** | [**List[StaticParam]**](StaticParam.md) | Optional typed defaults accessible on the evaluator instance. Omit or pass an empty array when the custom class does not read any static parameters.  | [optional] 

## Example

```python
from arize._generated.api_client.models.custom_code_config import CustomCodeConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CustomCodeConfig from a JSON string
custom_code_config_instance = CustomCodeConfig.from_json(json)
# print the JSON string representation of the object
print(CustomCodeConfig.to_json())

# convert the object into a dict
custom_code_config_dict = custom_code_config_instance.to_dict()
# create an instance of CustomCodeConfig from a dict
custom_code_config_from_dict = CustomCodeConfig.from_dict(custom_code_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


