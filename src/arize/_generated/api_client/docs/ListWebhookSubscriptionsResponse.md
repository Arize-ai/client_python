# ListWebhookSubscriptionsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**subscriptions** | [**List[WebhookSubscription]**](WebhookSubscription.md) | A list of webhook subscriptions, most recently created first | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_webhook_subscriptions_response import ListWebhookSubscriptionsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListWebhookSubscriptionsResponse from a JSON string
list_webhook_subscriptions_response_instance = ListWebhookSubscriptionsResponse.from_json(json)
# print the JSON string representation of the object
print(ListWebhookSubscriptionsResponse.to_json())

# convert the object into a dict
list_webhook_subscriptions_response_dict = list_webhook_subscriptions_response_instance.to_dict()
# create an instance of ListWebhookSubscriptionsResponse from a dict
list_webhook_subscriptions_response_from_dict = ListWebhookSubscriptionsResponse.from_dict(list_webhook_subscriptions_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


