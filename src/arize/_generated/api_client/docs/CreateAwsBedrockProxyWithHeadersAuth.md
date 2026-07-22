# CreateAwsBedrockProxyWithHeadersAuth

Create proxy auth. `base_url` is required. `headers` is write-only; names are returned as `header_names` on read.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** |  | 
**base_url** | **str** | Proxy URL requests are forwarded to (HTTPS). | 
**headers** | **Dict[str, str]** | Custom request headers sent to the proxy, as a name-to-value map. Write-only: values are never returned; names are exposed as &#x60;header_names&#x60; on read. Defaults to no headers. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_aws_bedrock_proxy_with_headers_auth import CreateAwsBedrockProxyWithHeadersAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAwsBedrockProxyWithHeadersAuth from a JSON string
create_aws_bedrock_proxy_with_headers_auth_instance = CreateAwsBedrockProxyWithHeadersAuth.from_json(json)
# print the JSON string representation of the object
print(CreateAwsBedrockProxyWithHeadersAuth.to_json())

# convert the object into a dict
create_aws_bedrock_proxy_with_headers_auth_dict = create_aws_bedrock_proxy_with_headers_auth_instance.to_dict()
# create an instance of CreateAwsBedrockProxyWithHeadersAuth from a dict
create_aws_bedrock_proxy_with_headers_auth_from_dict = CreateAwsBedrockProxyWithHeadersAuth.from_dict(create_aws_bedrock_proxy_with_headers_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


