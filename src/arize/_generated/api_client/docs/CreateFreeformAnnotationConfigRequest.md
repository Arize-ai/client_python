# CreateFreeformAnnotationConfigRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new annotation config | 
**space_id** | **str** | ID of the space the annotation config will belong to | 
**annotation_config_type** | **str** | Discriminator value identifying a freeform annotation config. | 

## Example

```python
from arize._generated.api_client.models.create_freeform_annotation_config_request import CreateFreeformAnnotationConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateFreeformAnnotationConfigRequest from a JSON string
create_freeform_annotation_config_request_instance = CreateFreeformAnnotationConfigRequest.from_json(json)
# print the JSON string representation of the object
print(CreateFreeformAnnotationConfigRequest.to_json())

# convert the object into a dict
create_freeform_annotation_config_request_dict = create_freeform_annotation_config_request_instance.to_dict()
# create an instance of CreateFreeformAnnotationConfigRequest from a dict
create_freeform_annotation_config_request_from_dict = CreateFreeformAnnotationConfigRequest.from_dict(create_freeform_annotation_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


