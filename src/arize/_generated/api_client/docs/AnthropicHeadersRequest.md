# AnthropicHeadersRequest

Anthropic-specific headers in a write request (strict form of AnthropicHeaders)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**anthropic_beta** | **List[Optional[str]]** | Anthropic beta feature flags | [optional] 

## Example

```python
from arize._generated.api_client.models.anthropic_headers_request import AnthropicHeadersRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AnthropicHeadersRequest from a JSON string
anthropic_headers_request_instance = AnthropicHeadersRequest.from_json(json)
# print the JSON string representation of the object
print(AnthropicHeadersRequest.to_json())

# convert the object into a dict
anthropic_headers_request_dict = anthropic_headers_request_instance.to_dict()
# create an instance of AnthropicHeadersRequest from a dict
anthropic_headers_request_from_dict = AnthropicHeadersRequest.from_dict(anthropic_headers_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


