# CreateCategoricalAnnotationConfigRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new annotation config | 
**space_id** | **str** | ID of the space the annotation config will belong to | 
**annotation_config_type** | **str** | Discriminator value identifying a categorical annotation config. | 
**values** | [**List[CategoricalAnnotationValue]**](CategoricalAnnotationValue.md) | An array of categorical annotation values | 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) | Direction for optimization. Defaults to &#x60;NONE&#x60; when omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_categorical_annotation_config_request import CreateCategoricalAnnotationConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateCategoricalAnnotationConfigRequest from a JSON string
create_categorical_annotation_config_request_instance = CreateCategoricalAnnotationConfigRequest.from_json(json)
# print the JSON string representation of the object
print(CreateCategoricalAnnotationConfigRequest.to_json())

# convert the object into a dict
create_categorical_annotation_config_request_dict = create_categorical_annotation_config_request_instance.to_dict()
# create an instance of CreateCategoricalAnnotationConfigRequest from a dict
create_categorical_annotation_config_request_from_dict = CreateCategoricalAnnotationConfigRequest.from_dict(create_categorical_annotation_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


