# SamlDomainVerification


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**domain** | **str** | Email domain. | 
**validated** | **bool** | Whether ownership of this domain is verified. | 

## Example

```python
from arize._generated.api_client.models.saml_domain_verification import SamlDomainVerification

# TODO update the JSON string below
json = "{}"
# create an instance of SamlDomainVerification from a JSON string
saml_domain_verification_instance = SamlDomainVerification.from_json(json)
# print the JSON string representation of the object
print(SamlDomainVerification.to_json())

# convert the object into a dict
saml_domain_verification_dict = saml_domain_verification_instance.to_dict()
# create an instance of SamlDomainVerification from a dict
saml_domain_verification_from_dict = SamlDomainVerification.from_dict(saml_domain_verification_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


