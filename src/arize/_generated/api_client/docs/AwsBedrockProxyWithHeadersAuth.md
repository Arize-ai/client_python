# AwsBedrockProxyWithHeadersAuth

Proxy auth for AWS Bedrock: requests are forwarded to a proxy URL with custom headers. Header values are write-only; configured names are returned as `header_names`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying proxy auth. | 
**base_url** | **str** | Proxy URL requests are forwarded to. | 
**header_names** | **List[str]** | Names of the custom request headers configured on this integration. Empty when none are configured. Header values are write-only and never returned. | 

## Example

```python
from arize._generated.api_client.models.aws_bedrock_proxy_with_headers_auth import AwsBedrockProxyWithHeadersAuth

# TODO update the JSON string below
json = "{}"
# create an instance of AwsBedrockProxyWithHeadersAuth from a JSON string
aws_bedrock_proxy_with_headers_auth_instance = AwsBedrockProxyWithHeadersAuth.from_json(json)
# print the JSON string representation of the object
print(AwsBedrockProxyWithHeadersAuth.to_json())

# convert the object into a dict
aws_bedrock_proxy_with_headers_auth_dict = aws_bedrock_proxy_with_headers_auth_instance.to_dict()
# create an instance of AwsBedrockProxyWithHeadersAuth from a dict
aws_bedrock_proxy_with_headers_auth_from_dict = AwsBedrockProxyWithHeadersAuth.from_dict(aws_bedrock_proxy_with_headers_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


