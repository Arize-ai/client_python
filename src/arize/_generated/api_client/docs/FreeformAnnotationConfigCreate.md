# FreeformAnnotationConfigCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new annotation config | 
**space_id** | **str** | ID of the space the annotation config will belong to | 
**annotation_config_type** | **str** | The type of the annotation config | 

## Example

```python
from arize._generated.api_client.models.freeform_annotation_config_create import FreeformAnnotationConfigCreate

# TODO update the JSON string below
json = "{}"
# create an instance of FreeformAnnotationConfigCreate from a JSON string
freeform_annotation_config_create_instance = FreeformAnnotationConfigCreate.from_json(json)
# print the JSON string representation of the object
print(FreeformAnnotationConfigCreate.to_json())

# convert the object into a dict
freeform_annotation_config_create_dict = freeform_annotation_config_create_instance.to_dict()
# create an instance of FreeformAnnotationConfigCreate from a dict
freeform_annotation_config_create_from_dict = FreeformAnnotationConfigCreate.from_dict(freeform_annotation_config_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


