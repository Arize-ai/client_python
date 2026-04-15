# OrganizationsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the organization (must be unique within the account) | 
**description** | **str** | A brief description of the organization&#39;s purpose. Defaults to an empty string if omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.organizations_create_request import OrganizationsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationsCreateRequest from a JSON string
organizations_create_request_instance = OrganizationsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(OrganizationsCreateRequest.to_json())

# convert the object into a dict
organizations_create_request_dict = organizations_create_request_instance.to_dict()
# create an instance of OrganizationsCreateRequest from a dict
organizations_create_request_from_dict = OrganizationsCreateRequest.from_dict(organizations_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


