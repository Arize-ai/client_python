# ManagedCodeConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_granularity** | **str** | Data granularity level for evaluation. When omitted or null, no granularity filter is applied (span-level evaluation is used by default on the server).  | [optional] 
**query_filter** | **str** | Optional filter query over the chosen data granularity. When omitted or null, no filter is applied.  | [optional] 
**type** | **str** | Discriminator for managed (built-in) code evaluators | 
**name** | **str** | Eval column name. Must match ^[a-zA-Z0-9_\\s\\-&amp;()]+$ | 
**managed_evaluator** | [**ManagedCodeEvaluator**](ManagedCodeEvaluator.md) |  | 
**variables** | **List[str]** | Dataset columns or span attributes passed into the evaluator (order and count must match the managed evaluator&#39;s requirements).  | 
**static_params** | [**List[StaticParam]**](StaticParam.md) | Static parameters for the managed evaluator (see registry &#x60;args&#x60;). When omitted, the registry&#39;s required arguments must be satisfied by defaults on the evaluator class; otherwise validation fails with 400. If the registry has no args, omitting this field is equivalent to an empty list.  | [optional] 

## Example

```python
from arize._generated.api_client.models.managed_code_config import ManagedCodeConfig

# TODO update the JSON string below
json = "{}"
# create an instance of ManagedCodeConfig from a JSON string
managed_code_config_instance = ManagedCodeConfig.from_json(json)
# print the JSON string representation of the object
print(ManagedCodeConfig.to_json())

# convert the object into a dict
managed_code_config_dict = managed_code_config_instance.to_dict()
# create an instance of ManagedCodeConfig from a dict
managed_code_config_from_dict = ManagedCodeConfig.from_dict(managed_code_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


