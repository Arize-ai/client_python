# ServiceKeyBotUser


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Global ID of the bot user. | 
**name** | **str** | Display name of the bot user. | 
**account_role** | [**UserRoleAssignment**](UserRoleAssignment.md) | Account-level role assigned to the bot user. Always present — defaults are resolved server-side. In a list response (&#x60;GET /v2/api-keys&#x60;) for a bot user that could not be resolved, this is a &#x60;MEMBER&#x60; placeholder and &#x60;name&#x60; is empty.  | 
**organizations** | [**List[ServiceKeyBotUserOrgAssignment]**](ServiceKeyBotUserOrgAssignment.md) | Organization access assignments for the service account, each containing nested space assignments. Always empty in a list response (&#x60;GET /v2/api-keys&#x60;) filtered by &#x60;space_id&#x60;. Also empty in a list response for a bot user whose bindings could not be resolved (e.g. a stale or otherwise unrecoverable bot user). &#x60;CreatedServiceKeyBotUser&#x60; (used in the &#x60;POST /v2/api-keys&#x60; response) tightens this to at least one entry, since creating a service key always requires an organization.  | 

## Example

```python
from arize._generated.api_client.models.service_key_bot_user import ServiceKeyBotUser

# TODO update the JSON string below
json = "{}"
# create an instance of ServiceKeyBotUser from a JSON string
service_key_bot_user_instance = ServiceKeyBotUser.from_json(json)
# print the JSON string representation of the object
print(ServiceKeyBotUser.to_json())

# convert the object into a dict
service_key_bot_user_dict = service_key_bot_user_instance.to_dict()
# create an instance of ServiceKeyBotUser from a dict
service_key_bot_user_from_dict = ServiceKeyBotUser.from_dict(service_key_bot_user_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


