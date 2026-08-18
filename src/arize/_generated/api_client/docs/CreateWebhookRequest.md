# CreateWebhookRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organization_id** | **str** | The unique identifier of the organization to create the webhook in | 
**name** | **str** | Name of the webhook (must be unique within the organization) | 
**url** | **str** | The HTTPS endpoint events are delivered to | 
**description** | **str** | A brief description of the webhook&#39;s purpose. Defaults to an empty string if omitted. | [optional] 
**auth_type** | [**WebhookAuthType**](WebhookAuthType.md) | How deliveries from this webhook are authenticated. Defaults to &#x60;BEARER&#x60; if omitted, and cannot be changed after creation. For &#x60;HMAC_SHA256&#x60;, a signing secret is generated for you and returned once in the create response.  | [optional] 
**auth_token** | **str** | The complete &#x60;Authorization&#x60; header value sent with each delivery request, e.g. &#x60;Bearer my-token&#x60;. Sent verbatim — include the &#x60;Bearer &#x60; prefix if your endpoint expects one. Only valid when &#x60;auth_type&#x60; is &#x60;BEARER&#x60;. Write-only: never returned in any response.  | [optional] 
**timeout_ms** | **int** | How long a delivery request may run before it is abandoned, in milliseconds. Defaults to 30000 if omitted. | [optional] 
**headers** | **Dict[str, str]** | Custom HTTP headers sent with each delivery request, as a map of at most 20 header names to values. Header names must be valid HTTP header names; connection-management headers (e.g. &#x60;Host&#x60;, &#x60;Content-Length&#x60;) are rejected.  | [optional] 

## Example

```python
from arize._generated.api_client.models.create_webhook_request import CreateWebhookRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateWebhookRequest from a JSON string
create_webhook_request_instance = CreateWebhookRequest.from_json(json)
# print the JSON string representation of the object
print(CreateWebhookRequest.to_json())

# convert the object into a dict
create_webhook_request_dict = create_webhook_request_instance.to_dict()
# create an instance of CreateWebhookRequest from a dict
create_webhook_request_from_dict = CreateWebhookRequest.from_dict(create_webhook_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


