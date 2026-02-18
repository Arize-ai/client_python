# AnnotationConfigCreateBase

The base annotation config creation parameters

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new annotation config | 
**space_id** | **str** | ID of the space the annotation config will belong to | 

## Example

```python
from arize._generated.api_client.models.annotation_config_create_base import AnnotationConfigCreateBase

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationConfigCreateBase from a JSON string
annotation_config_create_base_instance = AnnotationConfigCreateBase.from_json(json)
# print the JSON string representation of the object
print(AnnotationConfigCreateBase.to_json())

# convert the object into a dict
annotation_config_create_base_dict = annotation_config_create_base_instance.to_dict()
# create an instance of AnnotationConfigCreateBase from a dict
annotation_config_create_base_from_dict = AnnotationConfigCreateBase.from_dict(annotation_config_create_base_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


