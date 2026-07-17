# AnthropicHeaders

Anthropic-specific headers

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**anthropic_beta** | **List[Optional[str]]** | Anthropic beta feature flags | [optional] 

## Example

```python
from arize._generated.api_client.models.anthropic_headers import AnthropicHeaders

# TODO update the JSON string below
json = "{}"
# create an instance of AnthropicHeaders from a JSON string
anthropic_headers_instance = AnthropicHeaders.from_json(json)
# print the JSON string representation of the object
print(AnthropicHeaders.to_json())

# convert the object into a dict
anthropic_headers_dict = anthropic_headers_instance.to_dict()
# create an instance of AnthropicHeaders from a dict
anthropic_headers_from_dict = AnthropicHeaders.from_dict(anthropic_headers_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


