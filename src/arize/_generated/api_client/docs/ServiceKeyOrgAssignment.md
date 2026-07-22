# ServiceKeyOrgAssignment


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**org_id** | **str** | ID of the organization to grant the service account access to. | 
**role** | [**OrganizationRoleAssignment**](OrganizationRoleAssignment.md) | Role for the bot user within this organization. Only predefined roles are supported at this level. Custom org roles (&#x60;{ \&quot;type\&quot;: \&quot;CUSTOM\&quot;, \&quot;id\&quot;: \&quot;...\&quot; }&#x60;) are not yet supported and return &#x60;422&#x60;. Support will be added in a future release. Defaults to &#x60;{ type: PREDEFINED, name: READ_ONLY }&#x60; when omitted. Must be at or below the caller&#39;s own effective organization role. The &#x60;ANNOTATOR&#x60; role is not valid for service keys and returns &#x60;422&#x60;.  | [optional] 
**spaces** | [**List[ServiceKeySpaceAssignment]**](ServiceKeySpaceAssignment.md) | Spaces within this organization the service account should have access to. Each entry specifies a space and optional role. All space IDs must belong to this organization.  | 

## Example

```python
from arize._generated.api_client.models.service_key_org_assignment import ServiceKeyOrgAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of ServiceKeyOrgAssignment from a JSON string
service_key_org_assignment_instance = ServiceKeyOrgAssignment.from_json(json)
# print the JSON string representation of the object
print(ServiceKeyOrgAssignment.to_json())

# convert the object into a dict
service_key_org_assignment_dict = service_key_org_assignment_instance.to_dict()
# create an instance of ServiceKeyOrgAssignment from a dict
service_key_org_assignment_from_dict = ServiceKeyOrgAssignment.from_dict(service_key_org_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


