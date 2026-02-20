# Space

A space is a container within an organization for grouping related projects, datasets, and experiments. Spaces enable team collaboration or isolated experimentation with role-based access control. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the space | 
**name** | **str** | Name of the space | 
**description** | **str** | A brief description of the space&#39;s purpose | 
**created_at** | **datetime** | Timestamp for when the space was created | 
**updated_at** | **datetime** | Timestamp for the last update of the space | 

## Example

```python
from arize._generated.api_client.models.space import Space

# TODO update the JSON string below
json = "{}"
# create an instance of Space from a JSON string
space_instance = Space.from_json(json)
# print the JSON string representation of the object
print(Space.to_json())

# convert the object into a dict
space_dict = space_instance.to_dict()
# create an instance of Space from a dict
space_from_dict = Space.from_dict(space_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


