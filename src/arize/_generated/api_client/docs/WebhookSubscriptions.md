# WebhookSubscriptions

The complete set of webhook subscriptions attached to a resource.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**subscriptions** | [**List[WebhookSubscription]**](WebhookSubscription.md) | The webhooks attached to the resource and the events each receives | 

## Example

```python
from arize._generated.api_client.models.webhook_subscriptions import WebhookSubscriptions

# TODO update the JSON string below
json = "{}"
# create an instance of WebhookSubscriptions from a JSON string
webhook_subscriptions_instance = WebhookSubscriptions.from_json(json)
# print the JSON string representation of the object
print(WebhookSubscriptions.to_json())

# convert the object into a dict
webhook_subscriptions_dict = webhook_subscriptions_instance.to_dict()
# create an instance of WebhookSubscriptions from a dict
webhook_subscriptions_from_dict = WebhookSubscriptions.from_dict(webhook_subscriptions_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


