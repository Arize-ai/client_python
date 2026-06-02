# InsertExperimentRunsBody


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_runs** | [**List[ExperimentRunCreate]**](ExperimentRunCreate.md) | Array of experiment run data to append to the experiment. Between 1 and 1000 runs per request. | 

## Example

```python
from arize._generated.api_client.models.insert_experiment_runs_body import InsertExperimentRunsBody

# TODO update the JSON string below
json = "{}"
# create an instance of InsertExperimentRunsBody from a JSON string
insert_experiment_runs_body_instance = InsertExperimentRunsBody.from_json(json)
# print the JSON string representation of the object
print(InsertExperimentRunsBody.to_json())

# convert the object into a dict
insert_experiment_runs_body_dict = insert_experiment_runs_body_instance.to_dict()
# create an instance of InsertExperimentRunsBody from a dict
insert_experiment_runs_body_from_dict = InsertExperimentRunsBody.from_dict(insert_experiment_runs_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


