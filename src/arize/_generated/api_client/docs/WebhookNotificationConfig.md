# WebhookNotificationConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**id** | **str** | The webhook to notify (base64 global ID). | 
**url** | **str** | The webhook endpoint URL. | [optional] 

## Example

```python
from arize._generated.api_client.models.webhook_notification_config import WebhookNotificationConfig

# TODO update the JSON string below
json = "{}"
# create an instance of WebhookNotificationConfig from a JSON string
webhook_notification_config_instance = WebhookNotificationConfig.from_json(json)
# print the JSON string representation of the object
print(WebhookNotificationConfig.to_json())

# convert the object into a dict
webhook_notification_config_dict = webhook_notification_config_instance.to_dict()
# create an instance of WebhookNotificationConfig from a dict
webhook_notification_config_from_dict = WebhookNotificationConfig.from_dict(webhook_notification_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


