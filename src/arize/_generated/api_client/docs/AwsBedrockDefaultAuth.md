# AwsBedrockDefaultAuth

Role-assumption auth for AWS Bedrock: Arize assumes the provided IAM role to call Bedrock. The role ARN and external ID are not secrets and are returned on read.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying role-assumption auth. | 
**role_arn** | **str** | AWS IAM role ARN Arize assumes for cross-account access. | 
**external_id** | **str** | External ID on the assume-role policy. Null when not set. | 
**base_url** | **str** | Custom Bedrock endpoint URL. Null when not set. | 

## Example

```python
from arize._generated.api_client.models.aws_bedrock_default_auth import AwsBedrockDefaultAuth

# TODO update the JSON string below
json = "{}"
# create an instance of AwsBedrockDefaultAuth from a JSON string
aws_bedrock_default_auth_instance = AwsBedrockDefaultAuth.from_json(json)
# print the JSON string representation of the object
print(AwsBedrockDefaultAuth.to_json())

# convert the object into a dict
aws_bedrock_default_auth_dict = aws_bedrock_default_auth_instance.to_dict()
# create an instance of AwsBedrockDefaultAuth from a dict
aws_bedrock_default_auth_from_dict = AwsBedrockDefaultAuth.from_dict(aws_bedrock_default_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


