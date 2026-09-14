# CreateWebhookSubscriptionRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**webhook_id** | **str** | The unique identifier of the webhook to deliver the event to. Must belong to the source&#39;s organization. | 
**source_type** | [**WebhookSourceType**](WebhookSourceType.md) | The kind of resource to attach the webhook to | 
**source_id** | **str** | The unique identifier of the prompt or evaluator to attach the webhook to | 
**event** | [**WebhookEventType**](WebhookEventType.md) | The event to deliver. Must belong to the source type. | 

## Example

```python
from arize._generated.api_client.models.create_webhook_subscription_request import CreateWebhookSubscriptionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateWebhookSubscriptionRequest from a JSON string
create_webhook_subscription_request_instance = CreateWebhookSubscriptionRequest.from_json(json)
# print the JSON string representation of the object
print(CreateWebhookSubscriptionRequest.to_json())

# convert the object into a dict
create_webhook_subscription_request_dict = create_webhook_subscription_request_instance.to_dict()
# create an instance of CreateWebhookSubscriptionRequest from a dict
create_webhook_subscription_request_from_dict = CreateWebhookSubscriptionRequest.from_dict(create_webhook_subscription_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


