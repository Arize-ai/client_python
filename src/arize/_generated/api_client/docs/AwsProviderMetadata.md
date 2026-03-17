# AwsProviderMetadata

AWS Bedrock provider metadata

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_arn** | **str** | AWS IAM role ARN for cross-account access | 
**external_id** | **str** | External ID for the assume-role policy | [optional] 

## Example

```python
from arize._generated.api_client.models.aws_provider_metadata import AwsProviderMetadata

# TODO update the JSON string below
json = "{}"
# create an instance of AwsProviderMetadata from a JSON string
aws_provider_metadata_instance = AwsProviderMetadata.from_json(json)
# print the JSON string representation of the object
print(AwsProviderMetadata.to_json())

# convert the object into a dict
aws_provider_metadata_dict = aws_provider_metadata_instance.to_dict()
# create an instance of AwsProviderMetadata from a dict
aws_provider_metadata_from_dict = AwsProviderMetadata.from_dict(aws_provider_metadata_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


