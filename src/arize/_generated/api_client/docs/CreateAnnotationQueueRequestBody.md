# CreateAnnotationQueueRequestBody


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The name of the annotation queue. Must be unique within the space for active queues. | 
**space_id** | **str** | The space ID that the annotation queue belongs to | 
**instructions** | **str** | Instructions for annotators working on this queue | [optional] 
**annotation_config_ids** | **List[str]** | IDs of annotation configs to associate with this queue. All configs must belong to the same space. | 
**annotator_emails** | **List[str]** | Email addresses of annotators to assign to the queue. Emails are resolved to user IDs server-side. | 
**assignment_method** | **str** | How records are assigned to annotators. Defaults to \&quot;all\&quot;. - &#x60;all&#x60;: Every annotator is assigned to every record. - &#x60;random&#x60;: Each record is randomly assigned to one annotator.  | [optional] [default to 'all']
**record_sources** | [**List[AnnotationQueueRecordInput]**](AnnotationQueueRecordInput.md) | Record sources to add to the annotation queue on creation. At most 2 record sources (projects or datasets) may be provided in a single create request. Additional records from other sources can be added after creation. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_annotation_queue_request_body import CreateAnnotationQueueRequestBody

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAnnotationQueueRequestBody from a JSON string
create_annotation_queue_request_body_instance = CreateAnnotationQueueRequestBody.from_json(json)
# print the JSON string representation of the object
print(CreateAnnotationQueueRequestBody.to_json())

# convert the object into a dict
create_annotation_queue_request_body_dict = create_annotation_queue_request_body_instance.to_dict()
# create an instance of CreateAnnotationQueueRequestBody from a dict
create_annotation_queue_request_body_from_dict = CreateAnnotationQueueRequestBody.from_dict(create_annotation_queue_request_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


