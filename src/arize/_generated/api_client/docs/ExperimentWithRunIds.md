# ExperimentWithRunIds


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the experiment | 
**name** | **str** | Name of the experiment | 
**dataset_id** | **str** | Unique identifier for the dataset this experiment belongs to | 
**dataset_version_id** | **str** | Unique identifier for the dataset version this experiment belongs to | 
**created_at** | **datetime** | Timestamp for when the experiment was created | 
**updated_at** | **datetime** | Timestamp for the last update of the experiment | 
**experiment_traces_project_id** | **str** | Unique identifier for the experiment traces project this experiment belongs to (if it exists) | [optional] 
**run_ids** | **List[str]** | IDs of the newly inserted experiment runs, in input order. | 

## Example

```python
from arize._generated.api_client.models.experiment_with_run_ids import ExperimentWithRunIds

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentWithRunIds from a JSON string
experiment_with_run_ids_instance = ExperimentWithRunIds.from_json(json)
# print the JSON string representation of the object
print(ExperimentWithRunIds.to_json())

# convert the object into a dict
experiment_with_run_ids_dict = experiment_with_run_ids_instance.to_dict()
# create an instance of ExperimentWithRunIds from a dict
experiment_with_run_ids_from_dict = ExperimentWithRunIds.from_dict(experiment_with_run_ids_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


