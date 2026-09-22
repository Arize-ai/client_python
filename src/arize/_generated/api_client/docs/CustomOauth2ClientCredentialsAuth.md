# CustomOauth2ClientCredentialsAuth

OAuth 2.0 client-credentials auth (RFC 6749 section 4.4). Arize exchanges the stored credentials at `token_url` for a short-lived bearer token and sends it as `Authorization: Bearer` on every request to the endpoint. The client secret is write-only and surfaces as `has_client_secret`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying OAuth 2.0 client-credentials auth. | 
**token_url** | **str** | Token endpoint Arize exchanges the client credentials at. | 
**client_id** | **str** | OAuth client ID. Not a secret, so it is returned on read. | 
**has_client_secret** | **bool** | Whether a client secret is configured (the secret itself is never returned). | 
**scopes** | **str** | Space-separated scopes requested at the token endpoint. Null when not set. | 
**audience** | **str** | Audience requested at the token endpoint. Null when not set. | 

## Example

```python
from arize._generated.api_client.models.custom_oauth2_client_credentials_auth import CustomOauth2ClientCredentialsAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CustomOauth2ClientCredentialsAuth from a JSON string
custom_oauth2_client_credentials_auth_instance = CustomOauth2ClientCredentialsAuth.from_json(json)
# print the JSON string representation of the object
print(CustomOauth2ClientCredentialsAuth.to_json())

# convert the object into a dict
custom_oauth2_client_credentials_auth_dict = custom_oauth2_client_credentials_auth_instance.to_dict()
# create an instance of CustomOauth2ClientCredentialsAuth from a dict
custom_oauth2_client_credentials_auth_from_dict = CustomOauth2ClientCredentialsAuth.from_dict(custom_oauth2_client_credentials_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


