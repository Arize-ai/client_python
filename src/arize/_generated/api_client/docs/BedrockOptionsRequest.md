# BedrockOptionsRequest

AWS Bedrock options in a write request (strict form of BedrockOptions)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**use_converse_endpoint** | **bool** | Whether to use the AWS Bedrock Converse endpoint. Defaults to &#x60;false&#x60;. | [optional] [default to False]

## Example

```python
from arize._generated.api_client.models.bedrock_options_request import BedrockOptionsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of BedrockOptionsRequest from a JSON string
bedrock_options_request_instance = BedrockOptionsRequest.from_json(json)
# print the JSON string representation of the object
print(BedrockOptionsRequest.to_json())

# convert the object into a dict
bedrock_options_request_dict = bedrock_options_request_instance.to_dict()
# create an instance of BedrockOptionsRequest from a dict
bedrock_options_request_from_dict = BedrockOptionsRequest.from_dict(bedrock_options_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


