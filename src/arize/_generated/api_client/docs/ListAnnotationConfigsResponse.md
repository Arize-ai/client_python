# ListAnnotationConfigsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotation_configs** | [**List[AnnotationConfig]**](AnnotationConfig.md) | A list of annotation configs | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_annotation_configs_response import ListAnnotationConfigsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListAnnotationConfigsResponse from a JSON string
list_annotation_configs_response_instance = ListAnnotationConfigsResponse.from_json(json)
# print the JSON string representation of the object
print(ListAnnotationConfigsResponse.to_json())

# convert the object into a dict
list_annotation_configs_response_dict = list_annotation_configs_response_instance.to_dict()
# create an instance of ListAnnotationConfigsResponse from a dict
list_annotation_configs_response_from_dict = ListAnnotationConfigsResponse.from_dict(list_annotation_configs_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


