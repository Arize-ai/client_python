# WebhookNotificationConfigInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this notification channel as a webhook. | 
**id** | **str** | The webhook to notify (base64 global ID). It must belong to the project&#39;s organization and is subscribed to every monitor event.  | 

## Example

```python
from arize._generated.api_client.models.webhook_notification_config_input import WebhookNotificationConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of WebhookNotificationConfigInput from a JSON string
webhook_notification_config_input_instance = WebhookNotificationConfigInput.from_json(json)
# print the JSON string representation of the object
print(WebhookNotificationConfigInput.to_json())

# convert the object into a dict
webhook_notification_config_input_dict = webhook_notification_config_input_instance.to_dict()
# create an instance of WebhookNotificationConfigInput from a dict
webhook_notification_config_input_from_dict = WebhookNotificationConfigInput.from_dict(webhook_notification_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


