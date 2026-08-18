# WebhookSubscriptionInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**webhook_id** | **str** | The unique identifier of the webhook to attach | 
**subscribed_events** | [**List[WebhookEventType]**](WebhookEventType.md) | The events to deliver to the webhook. Must contain at least one event valid for the resource type. | 

## Example

```python
from arize._generated.api_client.models.webhook_subscription_input import WebhookSubscriptionInput

# TODO update the JSON string below
json = "{}"
# create an instance of WebhookSubscriptionInput from a JSON string
webhook_subscription_input_instance = WebhookSubscriptionInput.from_json(json)
# print the JSON string representation of the object
print(WebhookSubscriptionInput.to_json())

# convert the object into a dict
webhook_subscription_input_dict = webhook_subscription_input_instance.to_dict()
# create an instance of WebhookSubscriptionInput from a dict
webhook_subscription_input_from_dict = WebhookSubscriptionInput.from_dict(webhook_subscription_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


