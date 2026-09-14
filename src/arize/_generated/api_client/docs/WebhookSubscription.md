# WebhookSubscription

A subscription delivers one event from one prompt or evaluator to one webhook. A webhook that should receive several events from the same resource has one subscription per event. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the subscription | 
**webhook_id** | **str** | The unique identifier of the webhook that receives the event | 
**source_type** | [**WebhookSourceType**](WebhookSourceType.md) | The kind of resource the subscription is attached to | 
**source_id** | **str** | The unique identifier of the prompt or evaluator the subscription is attached to | 
**event** | [**WebhookEventType**](WebhookEventType.md) | The event delivered to the webhook | 
**created_at** | **datetime** | Timestamp for when the subscription was created | 

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


