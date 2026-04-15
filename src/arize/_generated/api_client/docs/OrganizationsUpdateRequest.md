# OrganizationsUpdateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Updated name for the organization (must be unique within the account) | [optional] 
**description** | **str** | Updated description for the organization. Set to an empty string to clear it. | [optional] 

## Example

```python
from arize._generated.api_client.models.organizations_update_request import OrganizationsUpdateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationsUpdateRequest from a JSON string
organizations_update_request_instance = OrganizationsUpdateRequest.from_json(json)
# print the JSON string representation of the object
print(OrganizationsUpdateRequest.to_json())

# convert the object into a dict
organizations_update_request_dict = organizations_update_request_instance.to_dict()
# create an instance of OrganizationsUpdateRequest from a dict
organizations_update_request_from_dict = OrganizationsUpdateRequest.from_dict(organizations_update_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


