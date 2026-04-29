# CodeConfigCommon


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**data_granularity** | **str** | Data granularity level for evaluation. When omitted or null, no granularity filter is applied (span-level evaluation is used by default on the server).  | [optional] 
**query_filter** | **str** | Optional filter query over the chosen data granularity. When omitted or null, no filter is applied.  | [optional] 

## Example

```python
from arize._generated.api_client.models.code_config_common import CodeConfigCommon

# TODO update the JSON string below
json = "{}"
# create an instance of CodeConfigCommon from a JSON string
code_config_common_instance = CodeConfigCommon.from_json(json)
# print the JSON string representation of the object
print(CodeConfigCommon.to_json())

# convert the object into a dict
code_config_common_dict = code_config_common_instance.to_dict()
# create an instance of CodeConfigCommon from a dict
code_config_common_from_dict = CodeConfigCommon.from_dict(code_config_common_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


