# FreeformAnnotationConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the annotation config | 
**name** | **str** | The name of the annotation config | 
**created_at** | **datetime** | The timestamp for when the annotation config was created | 
**space_id** | **str** | The space id the annotation config belongs to | 
**type** | **str** | The type of the annotation config | 

## Example

```python
from arize._generated.api_client.models.freeform_annotation_config import FreeformAnnotationConfig

# TODO update the JSON string below
json = "{}"
# create an instance of FreeformAnnotationConfig from a JSON string
freeform_annotation_config_instance = FreeformAnnotationConfig.from_json(json)
# print the JSON string representation of the object
print(FreeformAnnotationConfig.to_json())

# convert the object into a dict
freeform_annotation_config_dict = freeform_annotation_config_instance.to_dict()
# create an instance of FreeformAnnotationConfig from a dict
freeform_annotation_config_from_dict = FreeformAnnotationConfig.from_dict(freeform_annotation_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


