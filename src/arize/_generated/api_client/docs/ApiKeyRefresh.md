# ApiKeyRefresh


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**expires_at** | **datetime** | Expiration timestamp for the refreshed key. If omitted, the refreshed key has no expiration (infinite lifetime).  | [optional] 

## Example

```python
from arize._generated.api_client.models.api_key_refresh import ApiKeyRefresh

# TODO update the JSON string below
json = "{}"
# create an instance of ApiKeyRefresh from a JSON string
api_key_refresh_instance = ApiKeyRefresh.from_json(json)
# print the JSON string representation of the object
print(ApiKeyRefresh.to_json())

# convert the object into a dict
api_key_refresh_dict = api_key_refresh_instance.to_dict()
# create an instance of ApiKeyRefresh from a dict
api_key_refresh_from_dict = ApiKeyRefresh.from_dict(api_key_refresh_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


