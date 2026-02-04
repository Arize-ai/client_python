# ProjectsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**projects** | [**List[Project]**](Project.md) | A list of projects | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.projects_list200_response import ProjectsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ProjectsList200Response from a JSON string
projects_list200_response_instance = ProjectsList200Response.from_json(json)
# print the JSON string representation of the object
print(ProjectsList200Response.to_json())

# convert the object into a dict
projects_list200_response_dict = projects_list200_response_instance.to_dict()
# create an instance of ProjectsList200Response from a dict
projects_list200_response_from_dict = ProjectsList200Response.from_dict(projects_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


