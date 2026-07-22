# AwsBedrockConfig

Config for an AWS Bedrock LLM integration. The model catalog is caller-controlled via `is_default_models_enabled` and `model_names`. Function/tool-calling settings do not apply to Bedrock and are omitted.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** | Discriminator identifying the AWS Bedrock provider. | 
**is_default_models_enabled** | **bool** | Whether Arize&#39;s default Bedrock model catalog is enabled. | 
**model_names** | **List[str]** | Custom model names configured on this integration. Empty when none. | 
**auth** | [**AwsBedrockAuth**](AwsBedrockAuth.md) |  | 

## Example

```python
from arize._generated.api_client.models.aws_bedrock_config import AwsBedrockConfig

# TODO update the JSON string below
json = "{}"
# create an instance of AwsBedrockConfig from a JSON string
aws_bedrock_config_instance = AwsBedrockConfig.from_json(json)
# print the JSON string representation of the object
print(AwsBedrockConfig.to_json())

# convert the object into a dict
aws_bedrock_config_dict = aws_bedrock_config_instance.to_dict()
# create an instance of AwsBedrockConfig from a dict
aws_bedrock_config_from_dict = AwsBedrockConfig.from_dict(aws_bedrock_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


