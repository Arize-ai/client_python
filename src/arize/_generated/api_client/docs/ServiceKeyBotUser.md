# ServiceKeyBotUser


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Global ID of the bot user. | 
**name** | **str** | Display name of the bot user. | 
**account_role** | [**UserRoleAssignment**](UserRoleAssignment.md) | Account-level role assigned to the bot user. Always present — defaults are resolved server-side. | 
**organizations** | [**List[ServiceKeyBotUserOrgAssignment]**](ServiceKeyBotUserOrgAssignment.md) | Organization access assignments for the service account, each containing nested space assignments. | 

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


