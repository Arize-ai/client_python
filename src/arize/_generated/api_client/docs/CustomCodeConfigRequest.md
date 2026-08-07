# CustomCodeConfigRequest

Custom (user-supplied Python) code evaluator configuration in a write request (strict form of CustomCodeConfig)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_granularity** | [**DataGranularity**](DataGranularity.md) | Data granularity level for evaluation. When omitted or null, no granularity filter is applied (span-level evaluation is used by default on the server).  | [optional] 
**query_filter** | **str** | Optional filter query over the chosen data granularity. When omitted or null, no filter is applied.  | [optional] 
**type** | **str** | Discriminator identifying this as a custom (user-supplied Python) code evaluator | 
**name** | **str** | Eval column name. Must match ^[a-zA-Z0-9_\\s\\-&amp;()]+$ | 
**code** | **str** | Python source defining the evaluator class | 
**imports** | **str** | Optional package import block prepended when running the evaluator | [optional] 
**variables** | **List[str]** | Dataset columns or span attributes mapped to evaluate() arguments | 
**static_params** | [**List[StaticParamRequest]**](StaticParamRequest.md) | Optional typed defaults accessible on the evaluator instance. Omit or pass an empty array when the custom class does not read any static parameters.  | [optional] 

## Example

```python
from arize._generated.api_client.models.custom_code_config_request import CustomCodeConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CustomCodeConfigRequest from a JSON string
custom_code_config_request_instance = CustomCodeConfigRequest.from_json(json)
# print the JSON string representation of the object
print(CustomCodeConfigRequest.to_json())

# convert the object into a dict
custom_code_config_request_dict = custom_code_config_request_instance.to_dict()
# create an instance of CustomCodeConfigRequest from a dict
custom_code_config_request_from_dict = CustomCodeConfigRequest.from_dict(custom_code_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


