# NotificationConfigInput

A destination notified on a triggered transition. `type` selects the channel.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this notification channel as an email address. | 
**email_address** | **str** | A valid email address, at most 254 characters, notified on a triggered transition. | 
**integration_id** | **str** | The alert integration to notify (base64 global ID). It must belong to the project&#39;s organization. | 
**id** | **str** | The webhook to notify (base64 global ID). It must belong to the project&#39;s organization and is subscribed to every monitor event.  | 

## Example

```python
from arize._generated.api_client.models.notification_config_input import NotificationConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of NotificationConfigInput from a JSON string
notification_config_input_instance = NotificationConfigInput.from_json(json)
# print the JSON string representation of the object
print(NotificationConfigInput.to_json())

# convert the object into a dict
notification_config_input_dict = notification_config_input_instance.to_dict()
# create an instance of NotificationConfigInput from a dict
notification_config_input_from_dict = NotificationConfigInput.from_dict(notification_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


