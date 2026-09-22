# CustomAuth

Custom endpoint auth settings, discriminated by `auth_type`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying OAuth 2.0 client-credentials auth. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 
**token_url** | **str** | Token endpoint Arize exchanges the client credentials at. | 
**client_id** | **str** | OAuth client ID. Not a secret, so it is returned on read. | 
**has_client_secret** | **bool** | Whether a client secret is configured (the secret itself is never returned). | 
**scopes** | **str** | Space-separated scopes requested at the token endpoint. Null when not set. | 
**audience** | **str** | Audience requested at the token endpoint. Null when not set. | 

## Example

```python
from arize._generated.api_client.models.custom_auth import CustomAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CustomAuth from a JSON string
custom_auth_instance = CustomAuth.from_json(json)
# print the JSON string representation of the object
print(CustomAuth.to_json())

# convert the object into a dict
custom_auth_dict = custom_auth_instance.to_dict()
# create an instance of CustomAuth from a dict
custom_auth_from_dict = CustomAuth.from_dict(custom_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


