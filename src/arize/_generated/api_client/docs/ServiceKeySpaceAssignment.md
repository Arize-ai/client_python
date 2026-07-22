# ServiceKeySpaceAssignment

Declares one space that the service key's service account should have access to.  The **space assignment** (`space_id`) identifies the target space. The **role assignment** (`role`) specifies the level of access within that space — either a named predefined role or a custom RBAC role identified by its ID. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**space_id** | **str** | ID of the space to grant the service account access to. | 
**role** | [**SpaceRoleAssignment**](SpaceRoleAssignment.md) | Role to assign the bot user within this space. A role assignment is either: - &#x60;{ \&quot;type\&quot;: \&quot;PREDEFINED\&quot;, \&quot;name\&quot;: \&quot;ADMIN\&quot; | \&quot;MEMBER\&quot; | \&quot;READ_ONLY\&quot; }&#x60; — a built-in space role - &#x60;{ \&quot;type\&quot;: \&quot;CUSTOM\&quot;, \&quot;id\&quot;: \&quot;&lt;encoded-role-id&gt;\&quot; }&#x60; — a custom RBAC role  Defaults to &#x60;{ \&quot;type\&quot;: \&quot;PREDEFINED\&quot;, \&quot;name\&quot;: \&quot;MEMBER\&quot; }&#x60; when omitted. Must be at or below the caller&#39;s own effective space role. The &#x60;ANNOTATOR&#x60; role is not valid for service keys and returns &#x60;422&#x60;.  | [optional] 

## Example

```python
from arize._generated.api_client.models.service_key_space_assignment import ServiceKeySpaceAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of ServiceKeySpaceAssignment from a JSON string
service_key_space_assignment_instance = ServiceKeySpaceAssignment.from_json(json)
# print the JSON string representation of the object
print(ServiceKeySpaceAssignment.to_json())

# convert the object into a dict
service_key_space_assignment_dict = service_key_space_assignment_instance.to_dict()
# create an instance of ServiceKeySpaceAssignment from a dict
service_key_space_assignment_from_dict = ServiceKeySpaceAssignment.from_dict(service_key_space_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


