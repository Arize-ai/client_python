# ListDashboardsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dashboards** | [**List[DashboardSummary]**](DashboardSummary.md) | A list of dashboards. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_dashboards_response import ListDashboardsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListDashboardsResponse from a JSON string
list_dashboards_response_instance = ListDashboardsResponse.from_json(json)
# print the JSON string representation of the object
print(ListDashboardsResponse.to_json())

# convert the object into a dict
list_dashboards_response_dict = list_dashboards_response_instance.to_dict()
# create an instance of ListDashboardsResponse from a dict
list_dashboards_response_from_dict = ListDashboardsResponse.from_dict(list_dashboards_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


