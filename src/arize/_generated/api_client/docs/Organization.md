# Organization

An organization is a top-level container within an account for grouping spaces and managing access control. Organizations enable team separation with role-based access control at the organization level. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the organization | 
**name** | **str** | Name of the organization | 
**description** | **str** | A brief description of the organization&#39;s purpose | 
**created_at** | **datetime** | Timestamp for when the organization was created | 

## Example

```python
from arize._generated.api_client.models.organization import Organization

# TODO update the JSON string below
json = "{}"
# create an instance of Organization from a JSON string
organization_instance = Organization.from_json(json)
# print the JSON string representation of the object
print(Organization.to_json())

# convert the object into a dict
organization_dict = organization_instance.to_dict()
# create an instance of Organization from a dict
organization_from_dict = Organization.from_dict(organization_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


