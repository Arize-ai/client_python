# ResponseFormatRequest

Response format configuration in a write request (strict form of ResponseFormat)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**ResponseFormatType**](ResponseFormatType.md) | The response format type. Defaults to &#x60;TEXT&#x60; if not specified. | [optional] 
**json_schema** | [**JsonSchemaConfigRequest**](JsonSchemaConfigRequest.md) | JSON schema configuration (when type is JSON_SCHEMA) | [optional] 

## Example

```python
from arize._generated.api_client.models.response_format_request import ResponseFormatRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ResponseFormatRequest from a JSON string
response_format_request_instance = ResponseFormatRequest.from_json(json)
# print the JSON string representation of the object
print(ResponseFormatRequest.to_json())

# convert the object into a dict
response_format_request_dict = response_format_request_instance.to_dict()
# create an instance of ResponseFormatRequest from a dict
response_format_request_from_dict = ResponseFormatRequest.from_dict(response_format_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


