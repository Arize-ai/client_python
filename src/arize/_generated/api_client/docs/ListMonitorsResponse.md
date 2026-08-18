# ListMonitorsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**monitors** | [**List[Monitor]**](Monitor.md) | A list of monitors. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_monitors_response import ListMonitorsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListMonitorsResponse from a JSON string
list_monitors_response_instance = ListMonitorsResponse.from_json(json)
# print the JSON string representation of the object
print(ListMonitorsResponse.to_json())

# convert the object into a dict
list_monitors_response_dict = list_monitors_response_instance.to_dict()
# create an instance of ListMonitorsResponse from a dict
list_monitors_response_from_dict = ListMonitorsResponse.from_dict(list_monitors_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


