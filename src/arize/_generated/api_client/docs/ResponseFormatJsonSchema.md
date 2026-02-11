# ResponseFormatJsonSchema

JSON schema configuration (when type is json_schema)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The name of the JSON schema | [optional] 
**description** | **str** | A description of the JSON schema | [optional] 
**var_schema** | **object** | The JSON schema object | [optional] 
**strict** | **bool** | Whether to enforce strict schema validation | [optional] 

## Example

```python
from arize._generated.api_client.models.response_format_json_schema import ResponseFormatJsonSchema

# TODO update the JSON string below
json = "{}"
# create an instance of ResponseFormatJsonSchema from a JSON string
response_format_json_schema_instance = ResponseFormatJsonSchema.from_json(json)
# print the JSON string representation of the object
print(ResponseFormatJsonSchema.to_json())

# convert the object into a dict
response_format_json_schema_dict = response_format_json_schema_instance.to_dict()
# create an instance of ResponseFormatJsonSchema from a dict
response_format_json_schema_from_dict = ResponseFormatJsonSchema.from_dict(response_format_json_schema_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


