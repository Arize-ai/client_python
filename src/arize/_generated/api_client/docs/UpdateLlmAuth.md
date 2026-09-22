# UpdateLlmAuth

Replacement auth settings for the providers that model credentials as an auth block: `AWS_BEDROCK` and `CUSTOM`. The block must match the stored provider; a Bedrock block on a `CUSTOM` integration (or the reverse) is rejected with 422. There is no discriminator here because `auth_type: DEFAULT` means different things for the two providers — role assumption for Bedrock, API-key auth for a custom endpoint — so the correct variant is determined by the stored provider rather than by `auth_type` alone.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying OAuth 2.0 client-credentials auth. | 
**role_arn** | **str** | AWS IAM role ARN Arize assumes for cross-account access. | 
**external_id** | **str** | External ID on the assume-role policy. Defaults to not set. | [optional] 
**base_url** | **str** | Proxy URL requests are forwarded to (HTTPS). | 
**api_key** | **str** | API key for the endpoint (write-only, never returned). | 
**headers** | **Dict[str, str]** | Custom request headers sent to the proxy, as a name-to-value map. Write-only: values are never returned; names are exposed as &#x60;header_names&#x60; on read. Defaults to no headers. The serialized header map must not exceed 8,175 bytes. | [optional] 
**token_url** | **str** | Token endpoint Arize exchanges the client credentials at (HTTPS; validated server-side and must resolve to a public address). | 
**client_id** | **str** | OAuth client ID. Not a secret, so it is returned on read. Must not contain a colon, which is ambiguous in HTTP Basic authentication. | 
**client_secret** | **str** | OAuth client secret (write-only, never returned). | 
**scopes** | **str** | Space-separated scopes to request at the token endpoint. Defaults to not set. | [optional] 
**audience** | **str** | Audience to request at the token endpoint, required by some authorization servers. Defaults to not set. | [optional] 

## Example

```python
from arize._generated.api_client.models.update_llm_auth import UpdateLlmAuth

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateLlmAuth from a JSON string
update_llm_auth_instance = UpdateLlmAuth.from_json(json)
# print the JSON string representation of the object
print(UpdateLlmAuth.to_json())

# convert the object into a dict
update_llm_auth_dict = update_llm_auth_instance.to_dict()
# create an instance of UpdateLlmAuth from a dict
update_llm_auth_from_dict = UpdateLlmAuth.from_dict(update_llm_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


