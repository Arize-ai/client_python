# CreateAwsBedrockDefaultAuth

Create role-assumption auth. `role_arn` is required.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** |  | 
**role_arn** | **str** | AWS IAM role ARN Arize assumes for cross-account access. | 
**external_id** | **str** | External ID on the assume-role policy. Defaults to not set. | [optional] 
**base_url** | **str** | Custom Bedrock endpoint URL. Defaults to the provider default endpoint. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_aws_bedrock_default_auth import CreateAwsBedrockDefaultAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAwsBedrockDefaultAuth from a JSON string
create_aws_bedrock_default_auth_instance = CreateAwsBedrockDefaultAuth.from_json(json)
# print the JSON string representation of the object
print(CreateAwsBedrockDefaultAuth.to_json())

# convert the object into a dict
create_aws_bedrock_default_auth_dict = create_aws_bedrock_default_auth_instance.to_dict()
# create an instance of CreateAwsBedrockDefaultAuth from a dict
create_aws_bedrock_default_auth_from_dict = CreateAwsBedrockDefaultAuth.from_dict(create_aws_bedrock_default_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


