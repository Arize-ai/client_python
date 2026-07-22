# ServiceKeyBotUserOrgAssignment


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**org_id** | **str** | ID of the organization the service account has access to. | 
**role** | [**OrganizationRoleAssignment**](OrganizationRoleAssignment.md) | Role assigned to the bot user within this organization. Always present — defaults are resolved server-side. | 
**spaces** | [**List[ServiceKeyBotUserSpaceAssignment]**](ServiceKeyBotUserSpaceAssignment.md) | Space assignments within this organization. Roles are always present in the response (defaults are resolved server-side). | 

## Example

```python
from arize._generated.api_client.models.service_key_bot_user_org_assignment import ServiceKeyBotUserOrgAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of ServiceKeyBotUserOrgAssignment from a JSON string
service_key_bot_user_org_assignment_instance = ServiceKeyBotUserOrgAssignment.from_json(json)
# print the JSON string representation of the object
print(ServiceKeyBotUserOrgAssignment.to_json())

# convert the object into a dict
service_key_bot_user_org_assignment_dict = service_key_bot_user_org_assignment_instance.to_dict()
# create an instance of ServiceKeyBotUserOrgAssignment from a dict
service_key_bot_user_org_assignment_from_dict = ServiceKeyBotUserOrgAssignment.from_dict(service_key_bot_user_org_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


