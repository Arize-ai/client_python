# ProjectListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**projects** | [**List[Project]**](Project.md) | A list of projects | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.project_list_response import ProjectListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ProjectListResponse from a JSON string
project_list_response_instance = ProjectListResponse.from_json(json)
# print the JSON string representation of the object
print(ProjectListResponse.to_json())

# convert the object into a dict
project_list_response_dict = project_list_response_instance.to_dict()
# create an instance of ProjectListResponse from a dict
project_list_response_from_dict = ProjectListResponse.from_dict(project_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


