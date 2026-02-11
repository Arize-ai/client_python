# ResponseFormat

Response format configuration

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**ResponseFormatType**](ResponseFormatType.md) |  | [optional] 
**json_schema** | [**ResponseFormatJsonSchema**](ResponseFormatJsonSchema.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.response_format import ResponseFormat

# TODO update the JSON string below
json = "{}"
# create an instance of ResponseFormat from a JSON string
response_format_instance = ResponseFormat.from_json(json)
# print the JSON string representation of the object
print(ResponseFormat.to_json())

# convert the object into a dict
response_format_dict = response_format_instance.to_dict()
# create an instance of ResponseFormat from a dict
response_format_from_dict = ResponseFormat.from_dict(response_format_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


