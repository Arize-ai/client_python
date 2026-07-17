# RefreshApiKeyRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**expires_at** | **datetime** | Expiration timestamp for the refreshed key. Required when the existing key has an expiry — omitting it would extend the key&#39;s lifetime to unbounded, which is rejected with 422. For an unbounded existing key, &#x60;expires_at&#x60; may be omitted (the refreshed key is also unbounded) or provided to add a specific expiry. The value must be no later than the old key&#39;s expiry — a request that would extend the key&#39;s lifetime is rejected with 422. To create a key with a longer lifetime, use &#x60;POST /v2/api-keys&#x60; to issue a new key rather than refreshing.  | [optional] 
**grace_period_seconds** | **int** | Grace period in seconds during which the old key remains valid after the refresh. When set, the old key&#39;s expiration is updated to &#x60;now + grace_period_seconds&#x60; instead of being immediately revoked — it expires naturally at the end of the window. If the old key already has an &#x60;expires_at&#x60; that is sooner than the grace window end, the shorter value is used (the grace period cannot extend a key&#39;s original lifetime). Defaults to 0 (immediate revocation). Maximum is 86400 (24 hours).  | [optional] 

## Example

```python
from arize._generated.api_client.models.refresh_api_key_request import RefreshApiKeyRequest

# TODO update the JSON string below
json = "{}"
# create an instance of RefreshApiKeyRequest from a JSON string
refresh_api_key_request_instance = RefreshApiKeyRequest.from_json(json)
# print the JSON string representation of the object
print(RefreshApiKeyRequest.to_json())

# convert the object into a dict
refresh_api_key_request_dict = refresh_api_key_request_instance.to_dict()
# create an instance of RefreshApiKeyRequest from a dict
refresh_api_key_request_from_dict = RefreshApiKeyRequest.from_dict(refresh_api_key_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


