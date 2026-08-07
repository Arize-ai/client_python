# CodeConfigRequest

Strict request form of CodeConfig. Discriminated union of ManagedCodeConfigRequest and CustomCodeConfigRequest. Use in write request bodies. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_granularity** | [**DataGranularity**](DataGranularity.md) | Data granularity level for evaluation. When omitted or null, no granularity filter is applied (span-level evaluation is used by default on the server).  | [optional] 
**query_filter** | **str** | Optional filter query over the chosen data granularity. When omitted or null, no filter is applied.  | [optional] 
**type** | **str** | Discriminator identifying this as a managed (built-in) code evaluator | 
**name** | **str** | Eval column name. Must match ^[a-zA-Z0-9_\\s\\-&amp;()]+$ | 
**managed_evaluator** | [**ManagedCodeEvaluator**](ManagedCodeEvaluator.md) |  | 
**variables** | **List[str]** | Dataset columns or span attributes mapped to evaluate() arguments | 
**static_params** | [**List[StaticParamRequest]**](StaticParamRequest.md) | Optional typed defaults accessible on the evaluator instance. Omit or pass an empty array when the custom class does not read any static parameters.  | [optional] 
**code** | **str** | Python source defining the evaluator class | 
**imports** | **str** | Optional package import block prepended when running the evaluator | [optional] 

## Example

```python
from arize._generated.api_client.models.code_config_request import CodeConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CodeConfigRequest from a JSON string
code_config_request_instance = CodeConfigRequest.from_json(json)
# print the JSON string representation of the object
print(CodeConfigRequest.to_json())

# convert the object into a dict
code_config_request_dict = code_config_request_instance.to_dict()
# create an instance of CodeConfigRequest from a dict
code_config_request_from_dict = CodeConfigRequest.from_dict(code_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


