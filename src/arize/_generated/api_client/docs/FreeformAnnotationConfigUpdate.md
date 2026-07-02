# FreeformAnnotationConfigUpdate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New name for the annotation config. Must be unique within the space. | [optional] 
**annotation_config_type** | **str** | Discriminator value identifying a freeform annotation config. The config &#x60;type&#x60; is immutable and must match the stored config&#39;s type.  | 

## Example

```python
from arize._generated.api_client.models.freeform_annotation_config_update import FreeformAnnotationConfigUpdate

# TODO update the JSON string below
json = "{}"
# create an instance of FreeformAnnotationConfigUpdate from a JSON string
freeform_annotation_config_update_instance = FreeformAnnotationConfigUpdate.from_json(json)
# print the JSON string representation of the object
print(FreeformAnnotationConfigUpdate.to_json())

# convert the object into a dict
freeform_annotation_config_update_dict = freeform_annotation_config_update_instance.to_dict()
# create an instance of FreeformAnnotationConfigUpdate from a dict
freeform_annotation_config_update_from_dict = FreeformAnnotationConfigUpdate.from_dict(freeform_annotation_config_update_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


