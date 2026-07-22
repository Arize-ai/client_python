# AwsBedrockBearerTokenAuth

Bearer-token auth for AWS Bedrock. The token surfaces as `has_api_key` on read; the token itself is never returned.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying bearer-token auth. | 
**has_api_key** | **bool** | Whether a bearer token is configured (the token itself is never returned). Always true for integrations created through this API; may be false for integrations created through the Arize UI without a token. | 
**base_url** | **str** | Custom Bedrock endpoint URL. Null when not set. | 

## Example

```python
from arize._generated.api_client.models.aws_bedrock_bearer_token_auth import AwsBedrockBearerTokenAuth

# TODO update the JSON string below
json = "{}"
# create an instance of AwsBedrockBearerTokenAuth from a JSON string
aws_bedrock_bearer_token_auth_instance = AwsBedrockBearerTokenAuth.from_json(json)
# print the JSON string representation of the object
print(AwsBedrockBearerTokenAuth.to_json())

# convert the object into a dict
aws_bedrock_bearer_token_auth_dict = aws_bedrock_bearer_token_auth_instance.to_dict()
# create an instance of AwsBedrockBearerTokenAuth from a dict
aws_bedrock_bearer_token_auth_from_dict = AwsBedrockBearerTokenAuth.from_dict(aws_bedrock_bearer_token_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


