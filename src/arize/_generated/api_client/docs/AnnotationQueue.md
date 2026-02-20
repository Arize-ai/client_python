# AnnotationQueue


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the annotation queue | 
**name** | **str** | The name of the annotation queue | 
**space_id** | **str** | The space id the annotation queue belongs to | 
**instructions** | **str** | The instructions for the annotation queue | [optional] 
**annotation_configs** | [**List[AnnotationConfig]**](AnnotationConfig.md) | The annotation configs associated with this queue | [optional] 
**created_at** | **datetime** | The timestamp for when the annotation queue was created | 
**updated_at** | **datetime** | The timestamp for when the annotation queue was last updated | 

## Example

```python
from arize._generated.api_client.models.annotation_queue import AnnotationQueue

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueue from a JSON string
annotation_queue_instance = AnnotationQueue.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueue.to_json())

# convert the object into a dict
annotation_queue_dict = annotation_queue_instance.to_dict()
# create an instance of AnnotationQueue from a dict
annotation_queue_from_dict = AnnotationQueue.from_dict(annotation_queue_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


