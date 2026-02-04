# ExperimentsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiments** | [**List[Experiment]**](Experiment.md) | A list of experiments | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.experiments_list200_response import ExperimentsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentsList200Response from a JSON string
experiments_list200_response_instance = ExperimentsList200Response.from_json(json)
# print the JSON string representation of the object
print(ExperimentsList200Response.to_json())

# convert the object into a dict
experiments_list200_response_dict = experiments_list200_response_instance.to_dict()
# create an instance of ExperimentsList200Response from a dict
experiments_list200_response_from_dict = ExperimentsList200Response.from_dict(experiments_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


