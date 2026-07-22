# CreateAwsBedrockBearerTokenAuth

Create bearer-token auth. `api_key` is required and write-only (never returned; surfaces as `has_api_key` on read).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** |  | 
**api_key** | **str** | Bearer token for Bedrock (write-only, never returned). | 
**base_url** | **str** | Custom Bedrock endpoint URL. Defaults to the provider default endpoint. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_aws_bedrock_bearer_token_auth import CreateAwsBedrockBearerTokenAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAwsBedrockBearerTokenAuth from a JSON string
create_aws_bedrock_bearer_token_auth_instance = CreateAwsBedrockBearerTokenAuth.from_json(json)
# print the JSON string representation of the object
print(CreateAwsBedrockBearerTokenAuth.to_json())

# convert the object into a dict
create_aws_bedrock_bearer_token_auth_dict = create_aws_bedrock_bearer_token_auth_instance.to_dict()
# create an instance of CreateAwsBedrockBearerTokenAuth from a dict
create_aws_bedrock_bearer_token_auth_from_dict = CreateAwsBedrockBearerTokenAuth.from_dict(create_aws_bedrock_bearer_token_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


