# CreateOrganizationRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the organization (must be unique within the account) | 
**description** | **str** | A brief description of the organization&#39;s purpose. Defaults to an empty string if omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_organization_request import CreateOrganizationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateOrganizationRequest from a JSON string
create_organization_request_instance = CreateOrganizationRequest.from_json(json)
# print the JSON string representation of the object
print(CreateOrganizationRequest.to_json())

# convert the object into a dict
create_organization_request_dict = create_organization_request_instance.to_dict()
# create an instance of CreateOrganizationRequest from a dict
create_organization_request_from_dict = CreateOrganizationRequest.from_dict(create_organization_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


