# CreateAnnotationConfigRequestBody


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotation_config_type** | [**AnnotationConfigType**](AnnotationConfigType.md) |  | 
**name** | **str** | Name of the new annotation config | 
**space_id** | **str** | ID of the space the annotation config will belong to | 
**minimum_score** | **float** | The minimum score value | 
**maximum_score** | **float** | The maximum score value | 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) |  | [optional] [default to OptimizationDirection.NONE]
**values** | [**List[CategoricalAnnotationValue]**](CategoricalAnnotationValue.md) | An array of categorical annotation values | 

## Example

```python
from arize._generated.api_client.models.create_annotation_config_request_body import CreateAnnotationConfigRequestBody

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAnnotationConfigRequestBody from a JSON string
create_annotation_config_request_body_instance = CreateAnnotationConfigRequestBody.from_json(json)
# print the JSON string representation of the object
print(CreateAnnotationConfigRequestBody.to_json())

# convert the object into a dict
create_annotation_config_request_body_dict = create_annotation_config_request_body_instance.to_dict()
# create an instance of CreateAnnotationConfigRequestBody from a dict
create_annotation_config_request_body_from_dict = CreateAnnotationConfigRequestBody.from_dict(create_annotation_config_request_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


