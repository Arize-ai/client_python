# SetWebhookSubscriptionsRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**subscriptions** | [**List[WebhookSubscriptionInput]**](WebhookSubscriptionInput.md) | The complete set of webhook subscriptions for the resource, with at most one entry per &#x60;webhook_id&#x60;. Replaces all existing subscriptions: subscriptions not included in the request are removed, and an empty array detaches every webhook from the resource. At most 200 webhooks may subscribe to the same event on a resource; requests that would exceed this limit are rejected.  | 

## Example

```python
from arize._generated.api_client.models.set_webhook_subscriptions_request import SetWebhookSubscriptionsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SetWebhookSubscriptionsRequest from a JSON string
set_webhook_subscriptions_request_instance = SetWebhookSubscriptionsRequest.from_json(json)
# print the JSON string representation of the object
print(SetWebhookSubscriptionsRequest.to_json())

# convert the object into a dict
set_webhook_subscriptions_request_dict = set_webhook_subscriptions_request_instance.to_dict()
# create an instance of SetWebhookSubscriptionsRequest from a dict
set_webhook_subscriptions_request_from_dict = SetWebhookSubscriptionsRequest.from_dict(set_webhook_subscriptions_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


