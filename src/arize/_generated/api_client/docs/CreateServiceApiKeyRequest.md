# CreateServiceApiKeyRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**key_type** | **str** | Must be &#x60;\&quot;SERVICE\&quot;&#x60;. | 
**name** | **str** | User-defined name for the API key. | 
**description** | **str** | Optional user-defined description for the API key. | [optional] 
**expires_at** | **datetime** | Optional expiration timestamp. If the account has no maximum API key lifetime, omitting this value creates a key that never expires. If a maximum is configured, an omitted value defaults to that maximum, and an explicit value exceeding it is rejected with 422.  | [optional] 
**account_role** | [**UserRoleAssignmentRequest**](UserRoleAssignmentRequest.md) | Account-level role for the bot user. Only predefined roles are supported at this level. Custom account roles (&#x60;{ \&quot;type\&quot;: \&quot;CUSTOM\&quot;, \&quot;id\&quot;: \&quot;...\&quot; }&#x60;) are not yet supported and return &#x60;422&#x60;. Support will be added in a future release. Defaults to &#x60;{ type: PREDEFINED, name: MEMBER }&#x60; when omitted. Must be at or below the caller&#39;s own account role. The &#x60;MEMBER_UI_ONLY&#x60; and &#x60;ANNOTATOR&#x60; roles are not valid for service keys and return &#x60;400&#x60;.  | [optional] 
**organizations** | [**List[ServiceKeyOrgAssignment]**](ServiceKeyOrgAssignment.md) | Organizations the service account should have access to. Each entry specifies an organization and the spaces within it. Must include at least one organization with at least one space. All spaces must belong to the organization they are listed under.  | 

## Example

```python
from arize._generated.api_client.models.create_service_api_key_request import CreateServiceApiKeyRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateServiceApiKeyRequest from a JSON string
create_service_api_key_request_instance = CreateServiceApiKeyRequest.from_json(json)
# print the JSON string representation of the object
print(CreateServiceApiKeyRequest.to_json())

# convert the object into a dict
create_service_api_key_request_dict = create_service_api_key_request_instance.to_dict()
# create an instance of CreateServiceApiKeyRequest from a dict
create_service_api_key_request_from_dict = CreateServiceApiKeyRequest.from_dict(create_service_api_key_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


