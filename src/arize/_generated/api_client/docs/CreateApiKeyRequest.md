# CreateApiKeyRequest

Request body for creating an API key. Set `key_type` to select the kind of key: - `USER` — authenticates as the creating user, inheriting their current permissions. - `SERVICE` — authenticates as a service account: a dedicated, automatically provisioned   identity with roles explicitly configured in the spaces you specify. Use this for   automation, CI/CD pipelines, or any workload that should run independently of a   specific user. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**key_type** | [**ApiKeyType**](ApiKeyType.md) | The type of key to create. Use &#x60;\&quot;USER\&quot;&#x60; for a personal key that authenticates as you with your current permissions. Use &#x60;\&quot;SERVICE\&quot;&#x60; for a key tied to a dedicated service account with its own explicitly configured access across one or more spaces.  | 
**name** | **str** | User-defined name for the API key. | 
**description** | **str** | Optional user-defined description for the API key. | [optional] 
**expires_at** | **datetime** | Optional expiration timestamp. If omitted the key never expires. | [optional] 
**account_role** | [**UserRoleAssignment**](UserRoleAssignment.md) | Account-level role for the bot user. Only predefined roles are supported at this level. Custom account roles (&#x60;{ \&quot;type\&quot;: \&quot;CUSTOM\&quot;, \&quot;id\&quot;: \&quot;...\&quot; }&#x60;) are not yet supported and return &#x60;422&#x60;. Support will be added in a future release. Defaults to &#x60;{ type: PREDEFINED, name: MEMBER }&#x60; when omitted. Must be at or below the caller&#39;s own account role. The &#x60;ANNOTATOR&#x60; role is not valid for service keys and returns &#x60;422&#x60;.  | [optional] 
**organizations** | [**List[ServiceKeyOrgAssignment]**](ServiceKeyOrgAssignment.md) | Organizations the service account should have access to. Each entry specifies an organization and the spaces within it. Must include at least one organization with at least one space. All spaces must belong to the organization they are listed under.  | 

## Example

```python
from arize._generated.api_client.models.create_api_key_request import CreateApiKeyRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateApiKeyRequest from a JSON string
create_api_key_request_instance = CreateApiKeyRequest.from_json(json)
# print the JSON string representation of the object
print(CreateApiKeyRequest.to_json())

# convert the object into a dict
create_api_key_request_dict = create_api_key_request_instance.to_dict()
# create an instance of CreateApiKeyRequest from a dict
create_api_key_request_from_dict = CreateApiKeyRequest.from_dict(create_api_key_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


