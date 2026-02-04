# Project

A project represents an LLM application and serves as the primary container for observability data. Each project collects traces and spans that capture the execution flow of your application, enabling you to debug issues, monitor latency, and analyze token usage. Projects belong to a space and provide a centralized view of your application's performance. Use projects to organize related traces, run experiments against datasets, and track improvements over time. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The project ID | 
**name** | **str** | The project name | 
**space_id** | **str** | The space ID the project belongs to | 
**created_at** | **datetime** | When the project was created | 

## Example

```python
from arize._generated.api_client.models.project import Project

# TODO update the JSON string below
json = "{}"
# create an instance of Project from a JSON string
project_instance = Project.from_json(json)
# print the JSON string representation of the object
print(Project.to_json())

# convert the object into a dict
project_dict = project_instance.to_dict()
# create an instance of Project from a dict
project_from_dict = Project.from_dict(project_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


