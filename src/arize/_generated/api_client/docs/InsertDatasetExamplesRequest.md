# InsertDatasetExamplesRequest

Examples to append (insert) to a dataset version, with auto-generated IDs.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**examples** | **List[Dict[str, object]]** | Array of examples to append to the dataset version | 

## Example

```python
from arize._generated.api_client.models.insert_dataset_examples_request import InsertDatasetExamplesRequest

# TODO update the JSON string below
json = "{}"
# create an instance of InsertDatasetExamplesRequest from a JSON string
insert_dataset_examples_request_instance = InsertDatasetExamplesRequest.from_json(json)
# print the JSON string representation of the object
print(InsertDatasetExamplesRequest.to_json())

# convert the object into a dict
insert_dataset_examples_request_dict = insert_dataset_examples_request_instance.to_dict()
# create an instance of InsertDatasetExamplesRequest from a dict
insert_dataset_examples_request_from_dict = InsertDatasetExamplesRequest.from_dict(insert_dataset_examples_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


