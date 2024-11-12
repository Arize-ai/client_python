batch logger
============
Use this to log evaluations or spans to Arize in bulk. Read our quickstart guide for `logging evaluations to Arize <https://docs.arize.com/arize/llm-evaluation-and-annotations/catching-hallucinations/log-evaluations-to-arize>`_. 

To use in your code, import the following:

``from arize.pandas.logger import Client``

client
^^^^^^
.. automodule:: arize.pandas.logger
   :members:
   :exclude-members: __init__

types
^^^^^^
.. automodule:: arize.utils.types
   :exclude-members: LLMConfigColumnNames, LLMRunMetadata, LLMRunMetadataColumnNames, PromptTemplateColumnNames, add_to_column_count_dictionary, count_characters_raw_data, is_array_of, is_dict_of, is_iterable_of, is_json_str, is_list_of