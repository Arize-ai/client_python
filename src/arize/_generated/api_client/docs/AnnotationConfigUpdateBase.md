# AnnotationConfigUpdateBase

The base annotation config update parameters

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New name for the annotation config. Must be unique within the space. | [optional] 

## Example

```python
from arize._generated.api_client.models.annotation_config_update_base import AnnotationConfigUpdateBase

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationConfigUpdateBase from a JSON string
annotation_config_update_base_instance = AnnotationConfigUpdateBase.from_json(json)
# print the JSON string representation of the object
print(AnnotationConfigUpdateBase.to_json())

# convert the object into a dict
annotation_config_update_base_dict = annotation_config_update_base_instance.to_dict()
# create an instance of AnnotationConfigUpdateBase from a dict
annotation_config_update_base_from_dict = AnnotationConfigUpdateBase.from_dict(annotation_config_update_base_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


