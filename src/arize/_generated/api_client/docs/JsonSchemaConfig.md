# JsonSchemaConfig

JSON schema configuration (when type is JSON_SCHEMA)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The name of the JSON schema | [optional] 
**description** | **str** | A description of the JSON schema | [optional] 
**var_schema** | **object** | The JSON schema object | [optional] 
**strict** | **bool** | Whether to enforce strict schema validation. Defaults to &#x60;false&#x60;. | [optional] [default to False]

## Example

```python
from arize._generated.api_client.models.json_schema_config import JsonSchemaConfig

# TODO update the JSON string below
json = "{}"
# create an instance of JsonSchemaConfig from a JSON string
json_schema_config_instance = JsonSchemaConfig.from_json(json)
# print the JSON string representation of the object
print(JsonSchemaConfig.to_json())

# convert the object into a dict
json_schema_config_dict = json_schema_config_instance.to_dict()
# create an instance of JsonSchemaConfig from a dict
json_schema_config_from_dict = JsonSchemaConfig.from_dict(json_schema_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


