# ProjectsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the project (must be unique within the space) | 
**space_id** | **str** | ID of the space to create the project in | 

## Example

```python
from arize._generated.api_client.models.projects_create_request import ProjectsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ProjectsCreateRequest from a JSON string
projects_create_request_instance = ProjectsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(ProjectsCreateRequest.to_json())

# convert the object into a dict
projects_create_request_dict = projects_create_request_instance.to_dict()
# create an instance of ProjectsCreateRequest from a dict
projects_create_request_from_dict = ProjectsCreateRequest.from_dict(projects_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


