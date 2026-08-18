# ListWebhooksResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**webhooks** | [**List[Webhook]**](Webhook.md) | A list of webhooks | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_webhooks_response import ListWebhooksResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListWebhooksResponse from a JSON string
list_webhooks_response_instance = ListWebhooksResponse.from_json(json)
# print the JSON string representation of the object
print(ListWebhooksResponse.to_json())

# convert the object into a dict
list_webhooks_response_dict = list_webhooks_response_instance.to_dict()
# create an instance of ListWebhooksResponse from a dict
list_webhooks_response_from_dict = ListWebhooksResponse.from_dict(list_webhooks_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


