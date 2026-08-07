# AwsProviderMetadataRequest

AWS Bedrock provider metadata in a write request (strict form of AwsProviderMetadata).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**AwsProviderMetadataKind**](AwsProviderMetadataKind.md) |  | 
**role_arn** | **str** | AWS IAM role ARN for cross-account access | 
**external_id** | **str** | External ID for the assume-role policy | [optional] 

## Example

```python
from arize._generated.api_client.models.aws_provider_metadata_request import AwsProviderMetadataRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AwsProviderMetadataRequest from a JSON string
aws_provider_metadata_request_instance = AwsProviderMetadataRequest.from_json(json)
# print the JSON string representation of the object
print(AwsProviderMetadataRequest.to_json())

# convert the object into a dict
aws_provider_metadata_request_dict = aws_provider_metadata_request_instance.to_dict()
# create an instance of AwsProviderMetadataRequest from a dict
aws_provider_metadata_request_from_dict = AwsProviderMetadataRequest.from_dict(aws_provider_metadata_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


