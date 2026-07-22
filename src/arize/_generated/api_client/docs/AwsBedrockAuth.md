# AwsBedrockAuth

AWS Bedrock auth settings, discriminated by `auth_type`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying proxy auth. | 
**role_arn** | **str** | AWS IAM role ARN Arize assumes for cross-account access. | 
**external_id** | **str** | External ID on the assume-role policy. Null when not set. | 
**base_url** | **str** | Proxy URL requests are forwarded to. | 
**has_api_key** | **bool** | Whether a bearer token is configured (the token itself is never returned). Always true for integrations created through this API; may be false for integrations created through the Arize UI without a token. | 
**header_names** | **List[str]** | Names of the custom request headers configured on this integration. Empty when none are configured. Header values are write-only and never returned. | 

## Example

```python
from arize._generated.api_client.models.aws_bedrock_auth import AwsBedrockAuth

# TODO update the JSON string below
json = "{}"
# create an instance of AwsBedrockAuth from a JSON string
aws_bedrock_auth_instance = AwsBedrockAuth.from_json(json)
# print the JSON string representation of the object
print(AwsBedrockAuth.to_json())

# convert the object into a dict
aws_bedrock_auth_dict = aws_bedrock_auth_instance.to_dict()
# create an instance of AwsBedrockAuth from a dict
aws_bedrock_auth_from_dict = AwsBedrockAuth.from_dict(aws_bedrock_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


