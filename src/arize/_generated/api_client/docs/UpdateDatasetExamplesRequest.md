# UpdateDatasetExamplesRequest

Examples to update by ID matching, optionally into a new version.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**examples** | [**List[UpdateDatasetExampleInput]**](UpdateDatasetExampleInput.md) | Array of examples with &#39;id&#39; field for matching and updating existing records | 
**new_version** | **str** | Name for the new version. If provided (non-empty), creates a new version with that name.  If omitted or empty, updates the existing version in-place.  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_dataset_examples_request import UpdateDatasetExamplesRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateDatasetExamplesRequest from a JSON string
update_dataset_examples_request_instance = UpdateDatasetExamplesRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateDatasetExamplesRequest.to_json())

# convert the object into a dict
update_dataset_examples_request_dict = update_dataset_examples_request_instance.to_dict()
# create an instance of UpdateDatasetExamplesRequest from a dict
update_dataset_examples_request_from_dict = UpdateDatasetExamplesRequest.from_dict(update_dataset_examples_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


