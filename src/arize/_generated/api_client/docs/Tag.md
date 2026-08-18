# Tag

A tag is a reusable label defined once per space and attached to many resources across the platform, so the same vocabulary can be applied to projects, datasets, prompts, and more.  Tags are shared. Renaming a tag changes it everywhere it appears, and deleting a tag detaches it from every resource it was attached to. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier of the tag | 
**name** | **str** | The tag name. Unique within the space, compared case-insensitively. | 
**description** | **str** | Free-form description of what the tag is for. &#x60;null&#x60; when no description has been set.  | [optional] 
**color** | [**TagColor**](TagColor.md) | Display color for the tag. &#x60;null&#x60; when no color has been assigned, which clients render with a neutral treatment.  | [optional] 
**space_id** | **str** | The unique identifier of the space the tag belongs to | 
**created_at** | **datetime** | When the tag was created | 
**updated_at** | **datetime** | When the tag was last modified. Equal to &#x60;created_at&#x60; until the tag is updated.  | 

## Example

```python
from arize._generated.api_client.models.tag import Tag

# TODO update the JSON string below
json = "{}"
# create an instance of Tag from a JSON string
tag_instance = Tag.from_json(json)
# print the JSON string representation of the object
print(Tag.to_json())

# convert the object into a dict
tag_dict = tag_instance.to_dict()
# create an instance of Tag from a dict
tag_from_dict = Tag.from_dict(tag_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


