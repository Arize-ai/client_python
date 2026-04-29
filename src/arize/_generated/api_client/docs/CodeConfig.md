# CodeConfig

Discriminated union representing either a managed (built-in) or custom (user-supplied Python) code evaluator configuration, resolved by the nested `type` field (`managed` -> `ManagedCodeConfig`, `custom` -> `CustomCodeConfig`). This inner `type` is independent of the parent evaluator version's `type` (which is always `code` here). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_granularity** | **str** | Data granularity level for evaluation. When omitted or null, no granularity filter is applied (span-level evaluation is used by default on the server).  | [optional] 
**query_filter** | **str** | Optional filter query over the chosen data granularity. When omitted or null, no filter is applied.  | [optional] 
**type** | **str** | Discriminator for managed (built-in) code evaluators | 
**name** | **str** | Eval column name. Must match ^[a-zA-Z0-9_\\s\\-&amp;()]+$ | 
**managed_evaluator** | [**ManagedCodeEvaluator**](ManagedCodeEvaluator.md) |  | 
**variables** | **List[str]** | Dataset columns or span attributes mapped to evaluate() arguments | 
**static_params** | [**List[StaticParam]**](StaticParam.md) | Optional typed defaults accessible on the evaluator instance. Omit or pass an empty array when the custom class does not read any static parameters.  | [optional] 
**code** | **str** | Python source defining the evaluator class | 
**imports** | **str** | Optional package import block prepended when running the evaluator | [optional] 

## Example

```python
from arize._generated.api_client.models.code_config import CodeConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CodeConfig from a JSON string
code_config_instance = CodeConfig.from_json(json)
# print the JSON string representation of the object
print(CodeConfig.to_json())

# convert the object into a dict
code_config_dict = code_config_instance.to_dict()
# create an instance of CodeConfig from a dict
code_config_from_dict = CodeConfig.from_dict(code_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


