# UpdateFreeformAnnotationConfigRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New name for the annotation config. Must be unique within the space. | [optional] 
**annotation_config_type** | **str** | Discriminator value identifying a freeform annotation config. The config &#x60;type&#x60; is immutable and must match the stored config&#39;s type.  | 

## Example

```python
from arize._generated.api_client.models.update_freeform_annotation_config_request import UpdateFreeformAnnotationConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateFreeformAnnotationConfigRequest from a JSON string
update_freeform_annotation_config_request_instance = UpdateFreeformAnnotationConfigRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateFreeformAnnotationConfigRequest.to_json())

# convert the object into a dict
update_freeform_annotation_config_request_dict = update_freeform_annotation_config_request_instance.to_dict()
# create an instance of UpdateFreeformAnnotationConfigRequest from a dict
update_freeform_annotation_config_request_from_dict = UpdateFreeformAnnotationConfigRequest.from_dict(update_freeform_annotation_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


