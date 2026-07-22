# ServiceKeyBotUserSpaceAssignment


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**space_id** | **str** | ID of the space the service account has access to. | 
**role** | [**SpaceRoleAssignment**](SpaceRoleAssignment.md) | Role assigned to the bot user within this space. Always present — defaults are resolved server-side. | 

## Example

```python
from arize._generated.api_client.models.service_key_bot_user_space_assignment import ServiceKeyBotUserSpaceAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of ServiceKeyBotUserSpaceAssignment from a JSON string
service_key_bot_user_space_assignment_instance = ServiceKeyBotUserSpaceAssignment.from_json(json)
# print the JSON string representation of the object
print(ServiceKeyBotUserSpaceAssignment.to_json())

# convert the object into a dict
service_key_bot_user_space_assignment_dict = service_key_bot_user_space_assignment_instance.to_dict()
# create an instance of ServiceKeyBotUserSpaceAssignment from a dict
service_key_bot_user_space_assignment_from_dict = ServiceKeyBotUserSpaceAssignment.from_dict(service_key_bot_user_space_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


