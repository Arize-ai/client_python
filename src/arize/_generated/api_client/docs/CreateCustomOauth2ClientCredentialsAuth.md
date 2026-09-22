# CreateCustomOauth2ClientCredentialsAuth

Create OAuth 2.0 client-credentials auth (RFC 6749 section 4.4). Arize exchanges these credentials at `token_url` for a short-lived bearer token on each request to the endpoint. Mutually exclusive with a static `api_key`: supplying this auth block clears any stored API key.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying OAuth 2.0 client-credentials auth. | 
**token_url** | **str** | Token endpoint Arize exchanges the client credentials at (HTTPS; validated server-side and must resolve to a public address). | 
**client_id** | **str** | OAuth client ID. Not a secret, so it is returned on read. Must not contain a colon, which is ambiguous in HTTP Basic authentication. | 
**client_secret** | **str** | OAuth client secret (write-only, never returned). | 
**scopes** | **str** | Space-separated scopes to request at the token endpoint. Defaults to not set. | [optional] 
**audience** | **str** | Audience to request at the token endpoint, required by some authorization servers. Defaults to not set. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_custom_oauth2_client_credentials_auth import CreateCustomOauth2ClientCredentialsAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CreateCustomOauth2ClientCredentialsAuth from a JSON string
create_custom_oauth2_client_credentials_auth_instance = CreateCustomOauth2ClientCredentialsAuth.from_json(json)
# print the JSON string representation of the object
print(CreateCustomOauth2ClientCredentialsAuth.to_json())

# convert the object into a dict
create_custom_oauth2_client_credentials_auth_dict = create_custom_oauth2_client_credentials_auth_instance.to_dict()
# create an instance of CreateCustomOauth2ClientCredentialsAuth from a dict
create_custom_oauth2_client_credentials_auth_from_dict = CreateCustomOauth2ClientCredentialsAuth.from_dict(create_custom_oauth2_client_credentials_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


