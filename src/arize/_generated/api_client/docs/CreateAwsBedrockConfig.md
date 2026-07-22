# CreateAwsBedrockConfig

Create config for an AWS Bedrock LLM integration. `auth` selects one of three auth modes via `auth_type`. The integration must have at least one model available: enable `is_default_models_enabled` or provide at least one entry in `model_names`, otherwise the request is rejected with 422.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** |  | 
**auth** | [**CreateAwsBedrockAuth**](CreateAwsBedrockAuth.md) |  | 
**is_default_models_enabled** | **bool** | Enable Arize&#39;s default Bedrock model catalog. Defaults to false. | [optional] 
**model_names** | **List[str]** | Custom model names to make available. Defaults to none. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_aws_bedrock_config import CreateAwsBedrockConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAwsBedrockConfig from a JSON string
create_aws_bedrock_config_instance = CreateAwsBedrockConfig.from_json(json)
# print the JSON string representation of the object
print(CreateAwsBedrockConfig.to_json())

# convert the object into a dict
create_aws_bedrock_config_dict = create_aws_bedrock_config_instance.to_dict()
# create an instance of CreateAwsBedrockConfig from a dict
create_aws_bedrock_config_from_dict = CreateAwsBedrockConfig.from_dict(create_aws_bedrock_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


