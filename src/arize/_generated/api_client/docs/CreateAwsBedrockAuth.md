# CreateAwsBedrockAuth

AWS Bedrock auth settings for create and update, discriminated by `auth_type`. On PATCH this object replaces the stored auth settings wholesale (auth_type may change); omitted fields of the previous auth mode are cleared.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** |  | 
**role_arn** | **str** | AWS IAM role ARN Arize assumes for cross-account access. | 
**external_id** | **str** | External ID on the assume-role policy. Defaults to not set. | [optional] 
**base_url** | **str** | Proxy URL requests are forwarded to (HTTPS). | 
**api_key** | **str** | Bearer token for Bedrock (write-only, never returned). | 
**headers** | **Dict[str, str]** | Custom request headers sent to the proxy, as a name-to-value map. Write-only: values are never returned; names are exposed as &#x60;header_names&#x60; on read. Defaults to no headers. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_aws_bedrock_auth import CreateAwsBedrockAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAwsBedrockAuth from a JSON string
create_aws_bedrock_auth_instance = CreateAwsBedrockAuth.from_json(json)
# print the JSON string representation of the object
print(CreateAwsBedrockAuth.to_json())

# convert the object into a dict
create_aws_bedrock_auth_dict = create_aws_bedrock_auth_instance.to_dict()
# create an instance of CreateAwsBedrockAuth from a dict
create_aws_bedrock_auth_from_dict = CreateAwsBedrockAuth.from_dict(create_aws_bedrock_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


