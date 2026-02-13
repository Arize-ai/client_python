# AnnotationConfigBase


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the annotation config | 
**name** | **str** | The name of the annotation config | 
**created_at** | **datetime** | The timestamp for when the annotation config was created | 
**space_id** | **str** | The space id the annotation config belongs to | 

## Example

```python
from arize._generated.api_client.models.annotation_config_base import AnnotationConfigBase

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationConfigBase from a JSON string
annotation_config_base_instance = AnnotationConfigBase.from_json(json)
# print the JSON string representation of the object
print(AnnotationConfigBase.to_json())

# convert the object into a dict
annotation_config_base_dict = annotation_config_base_instance.to_dict()
# create an instance of AnnotationConfigBase from a dict
annotation_config_base_from_dict = AnnotationConfigBase.from_dict(annotation_config_base_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


