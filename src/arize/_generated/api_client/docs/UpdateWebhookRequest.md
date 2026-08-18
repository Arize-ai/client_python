# UpdateWebhookRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Updated name of the webhook (must be unique within the organization) | [optional] 
**description** | **str** | Updated description of the webhook. Set to &#x60;null&#x60; to clear it. | [optional] 
**url** | **str** | Updated HTTPS endpoint events are delivered to | [optional] 
**auth_token** | **str** | Replacement &#x60;Authorization&#x60; header value sent with each delivery request, e.g. &#x60;Bearer my-token&#x60;. Sent verbatim — include the &#x60;Bearer &#x60; prefix if your endpoint expects one. Only valid when the webhook&#39;s &#x60;auth_type&#x60; is &#x60;BEARER&#x60;. Write-only: never returned in any response.  | [optional] 
**timeout_ms** | **int** | Updated delivery timeout in milliseconds | [optional] 
**headers** | **Dict[str, str]** | Replacement custom HTTP headers, as a map of at most 20 header names to values. Replaces the whole header map; headers not included are removed.  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_webhook_request import UpdateWebhookRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateWebhookRequest from a JSON string
update_webhook_request_instance = UpdateWebhookRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateWebhookRequest.to_json())

# convert the object into a dict
update_webhook_request_dict = update_webhook_request_instance.to_dict()
# create an instance of UpdateWebhookRequest from a dict
update_webhook_request_from_dict = UpdateWebhookRequest.from_dict(update_webhook_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


