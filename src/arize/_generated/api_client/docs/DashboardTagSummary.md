# DashboardTagSummary


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The tag ID. | 
**name** | **str** | The tag name. | 

## Example

```python
from arize._generated.api_client.models.dashboard_tag_summary import DashboardTagSummary

# TODO update the JSON string below
json = "{}"
# create an instance of DashboardTagSummary from a JSON string
dashboard_tag_summary_instance = DashboardTagSummary.from_json(json)
# print the JSON string representation of the object
print(DashboardTagSummary.to_json())

# convert the object into a dict
dashboard_tag_summary_dict = dashboard_tag_summary_instance.to_dict()
# create an instance of DashboardTagSummary from a dict
dashboard_tag_summary_from_dict = DashboardTagSummary.from_dict(dashboard_tag_summary_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


