# CreateVertexAiConfig

Create config for a Google Vertex AI integration. No credentials are stored: Arize accesses Vertex through the configured GCP project. `project_id`, `location`, and `project_access_label` are all required.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** |  | 
**project_id** | **str** | GCP project ID Arize accesses Vertex through. | 
**location** | **str** | GCP region (e.g. us-central1). | 
**project_access_label** | **str** | Label used to verify Arize&#39;s access to the GCP project. | 

## Example

```python
from arize._generated.api_client.models.create_vertex_ai_config import CreateVertexAiConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateVertexAiConfig from a JSON string
create_vertex_ai_config_instance = CreateVertexAiConfig.from_json(json)
# print the JSON string representation of the object
print(CreateVertexAiConfig.to_json())

# convert the object into a dict
create_vertex_ai_config_dict = create_vertex_ai_config_instance.to_dict()
# create an instance of CreateVertexAiConfig from a dict
create_vertex_ai_config_from_dict = CreateVertexAiConfig.from_dict(create_vertex_ai_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


