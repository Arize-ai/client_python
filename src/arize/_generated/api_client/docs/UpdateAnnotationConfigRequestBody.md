# UpdateAnnotationConfigRequestBody


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotation_config_type** | [**AnnotationConfigType**](AnnotationConfigType.md) |  | 
**name** | **str** | New name for the annotation config. Must be unique within the space. | [optional] 
**minimum_score** | **float** | New minimum score value. | [optional] 
**maximum_score** | **float** | New maximum score value. | [optional] 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) | New optimization direction. | [optional] 
**values** | [**List[CategoricalAnnotationValue]**](CategoricalAnnotationValue.md) | The full replacement set of categorical annotation values (2–100 items).  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_annotation_config_request_body import UpdateAnnotationConfigRequestBody

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateAnnotationConfigRequestBody from a JSON string
update_annotation_config_request_body_instance = UpdateAnnotationConfigRequestBody.from_json(json)
# print the JSON string representation of the object
print(UpdateAnnotationConfigRequestBody.to_json())

# convert the object into a dict
update_annotation_config_request_body_dict = update_annotation_config_request_body_instance.to_dict()
# create an instance of UpdateAnnotationConfigRequestBody from a dict
update_annotation_config_request_body_from_dict = UpdateAnnotationConfigRequestBody.from_dict(update_annotation_config_request_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


