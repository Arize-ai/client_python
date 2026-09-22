# CreateCustomDefaultAuth

API-key auth. `api_key` is optional — a custom endpoint may require no credential at all. On PATCH this block replaces the stored auth wholesale, so omitting `api_key` clears any stored key rather than leaving it unchanged; send the key again to keep it.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying API-key auth. | 
**api_key** | **str** | API key for the endpoint (write-only, never returned). | [optional] 

## Example

```python
from arize._generated.api_client.models.create_custom_default_auth import CreateCustomDefaultAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CreateCustomDefaultAuth from a JSON string
create_custom_default_auth_instance = CreateCustomDefaultAuth.from_json(json)
# print the JSON string representation of the object
print(CreateCustomDefaultAuth.to_json())

# convert the object into a dict
create_custom_default_auth_dict = create_custom_default_auth_instance.to_dict()
# create an instance of CreateCustomDefaultAuth from a dict
create_custom_default_auth_from_dict = CreateCustomDefaultAuth.from_dict(create_custom_default_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


