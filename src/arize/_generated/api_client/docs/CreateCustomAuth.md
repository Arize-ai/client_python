# CreateCustomAuth

Custom endpoint auth settings for create and update, discriminated by `auth_type`. On PATCH this object replaces the stored auth settings wholesale (auth_type may change); credentials belonging to the auth mode being switched away from are cleared.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying OAuth 2.0 client-credentials auth. | 
**api_key** | **str** | API key for the endpoint (write-only, never returned). | [optional] 
**token_url** | **str** | Token endpoint Arize exchanges the client credentials at (HTTPS; validated server-side and must resolve to a public address). | 
**client_id** | **str** | OAuth client ID. Not a secret, so it is returned on read. Must not contain a colon, which is ambiguous in HTTP Basic authentication. | 
**client_secret** | **str** | OAuth client secret (write-only, never returned). | 
**scopes** | **str** | Space-separated scopes to request at the token endpoint. Defaults to not set. | [optional] 
**audience** | **str** | Audience to request at the token endpoint, required by some authorization servers. Defaults to not set. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_custom_auth import CreateCustomAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CreateCustomAuth from a JSON string
create_custom_auth_instance = CreateCustomAuth.from_json(json)
# print the JSON string representation of the object
print(CreateCustomAuth.to_json())

# convert the object into a dict
create_custom_auth_dict = create_custom_auth_instance.to_dict()
# create an instance of CreateCustomAuth from a dict
create_custom_auth_from_dict = CreateCustomAuth.from_dict(create_custom_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


