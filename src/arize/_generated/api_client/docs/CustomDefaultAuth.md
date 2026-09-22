# CustomDefaultAuth

API-key auth for a custom endpoint. The key surfaces as `has_api_key`; the key itself is never returned. A custom endpoint may require no credential at all, in which case `has_api_key` is false.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**auth_type** | **str** | Discriminator identifying API-key auth. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 

## Example

```python
from arize._generated.api_client.models.custom_default_auth import CustomDefaultAuth

# TODO update the JSON string below
json = "{}"
# create an instance of CustomDefaultAuth from a JSON string
custom_default_auth_instance = CustomDefaultAuth.from_json(json)
# print the JSON string representation of the object
print(CustomDefaultAuth.to_json())

# convert the object into a dict
custom_default_auth_dict = custom_default_auth_instance.to_dict()
# create an instance of CustomDefaultAuth from a dict
custom_default_auth_from_dict = CustomDefaultAuth.from_dict(custom_default_auth_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


