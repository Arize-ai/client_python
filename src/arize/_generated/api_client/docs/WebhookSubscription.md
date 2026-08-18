# WebhookSubscription

A webhook attached to a resource and the events it receives.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**webhook_id** | **str** | The unique identifier of the subscribed webhook | 
**webhook_name** | **str** | Name of the subscribed webhook | 
**subscribed_events** | [**List[WebhookEventType]**](WebhookEventType.md) | The events delivered to the webhook | 

## Example

```python
from arize._generated.api_client.models.webhook_subscription import WebhookSubscription

# TODO update the JSON string below
json = "{}"
# create an instance of WebhookSubscription from a JSON string
webhook_subscription_instance = WebhookSubscription.from_json(json)
# print the JSON string representation of the object
print(WebhookSubscription.to_json())

# convert the object into a dict
webhook_subscription_dict = webhook_subscription_instance.to_dict()
# create an instance of WebhookSubscription from a dict
webhook_subscription_from_dict = WebhookSubscription.from_dict(webhook_subscription_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


