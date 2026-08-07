# JsonSchemaConfigRequest

JSON schema configuration in a write request (strict form of JsonSchemaConfig)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The name of the JSON schema | [optional] 
**description** | **str** | A description of the JSON schema | [optional] 
**var_schema** | **object** | The JSON schema object | [optional] 
**strict** | **bool** | Whether to enforce strict schema validation. Defaults to &#x60;false&#x60;. | [optional] [default to False]

## Example

```python
from arize._generated.api_client.models.json_schema_config_request import JsonSchemaConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of JsonSchemaConfigRequest from a JSON string
json_schema_config_request_instance = JsonSchemaConfigRequest.from_json(json)
# print the JSON string representation of the object
print(JsonSchemaConfigRequest.to_json())

# convert the object into a dict
json_schema_config_request_dict = json_schema_config_request_instance.to_dict()
# create an instance of JsonSchemaConfigRequest from a dict
json_schema_config_request_from_dict = JsonSchemaConfigRequest.from_dict(json_schema_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


