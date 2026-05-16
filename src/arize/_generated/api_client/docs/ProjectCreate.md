# ProjectCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the project (must be unique within the space) | 
**space_id** | **str** | ID of the space to create the project in | 

## Example

```python
from arize._generated.api_client.models.project_create import ProjectCreate

# TODO update the JSON string below
json = "{}"
# create an instance of ProjectCreate from a JSON string
project_create_instance = ProjectCreate.from_json(json)
# print the JSON string representation of the object
print(ProjectCreate.to_json())

# convert the object into a dict
project_create_dict = project_create_instance.to_dict()
# create an instance of ProjectCreate from a dict
project_create_from_dict = ProjectCreate.from_dict(project_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


