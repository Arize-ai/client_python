# DatasetsExamplesUpdateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**examples** | [**List[DatasetExampleUpdate]**](DatasetExampleUpdate.md) | Array of examples with &#39;id&#39; field for matching and updating existing records | 
**new_version** | **str** | Name for the new version. If provided (non-empty), creates a new version with that name.  If omitted or empty, updates the existing version in-place.  | [optional] 

## Example

```python
from arize._generated.api_client.models.datasets_examples_update_request import DatasetsExamplesUpdateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetsExamplesUpdateRequest from a JSON string
datasets_examples_update_request_instance = DatasetsExamplesUpdateRequest.from_json(json)
# print the JSON string representation of the object
print(DatasetsExamplesUpdateRequest.to_json())

# convert the object into a dict
datasets_examples_update_request_dict = datasets_examples_update_request_instance.to_dict()
# create an instance of DatasetsExamplesUpdateRequest from a dict
datasets_examples_update_request_from_dict = DatasetsExamplesUpdateRequest.from_dict(datasets_examples_update_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


