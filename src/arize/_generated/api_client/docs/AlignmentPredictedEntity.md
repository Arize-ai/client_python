# AlignmentPredictedEntity

The entity whose judge labels are compared against the human annotations.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the entity that produced the judge labels (base64). An experiment identifier when &#x60;entity_type&#x60; is &#x60;EXPERIMENT&#x60;, or a dataset version identifier when &#x60;entity_type&#x60; is &#x60;DATASET_VERSION&#x60;.  | 
**entity_type** | [**AlignmentPredictedEntityType**](AlignmentPredictedEntityType.md) |  | 

## Example

```python
from arize._generated.api_client.models.alignment_predicted_entity import AlignmentPredictedEntity

# TODO update the JSON string below
json = "{}"
# create an instance of AlignmentPredictedEntity from a JSON string
alignment_predicted_entity_instance = AlignmentPredictedEntity.from_json(json)
# print the JSON string representation of the object
print(AlignmentPredictedEntity.to_json())

# convert the object into a dict
alignment_predicted_entity_dict = alignment_predicted_entity_instance.to_dict()
# create an instance of AlignmentPredictedEntity from a dict
alignment_predicted_entity_from_dict = AlignmentPredictedEntity.from_dict(alignment_predicted_entity_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


