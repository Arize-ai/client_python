# AnnotatorUser

A user assigned as an annotator, identified by ID and email.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the user | 
**email** | **str** | An email address | 

## Example

```python
from arize._generated.api_client.models.annotator_user import AnnotatorUser

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotatorUser from a JSON string
annotator_user_instance = AnnotatorUser.from_json(json)
# print the JSON string representation of the object
print(AnnotatorUser.to_json())

# convert the object into a dict
annotator_user_dict = annotator_user_instance.to_dict()
# create an instance of AnnotatorUser from a dict
annotator_user_from_dict = AnnotatorUser.from_dict(annotator_user_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


