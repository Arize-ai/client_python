types
^^^^^
These are the classes used across the experiment functions.

To import types, use the following:
``from arize.experimental.datasets.experiments.types import ...``

.. autoclass:: arize.experimental.datasets.experiments.types.Example
   :exclude-members: id, updated_at, input, output, metadata, dataset_row, from_dict

.. autoclass:: arize.experimental.datasets.experiments.types.EvaluationResult
   :exclude-members: score, label, explanation, metadata, from_dict

.. autoclass:: arize.experimental.datasets.experiments.types.ExperimentRun
   :exclude-members: start_time, end_time, experiment_id, dataset_example_id, repetition_number, output, error, id, trace_id, from_dict

.. autoclass:: arize.experimental.datasets.experiments.types.ExperimentEvaluationRun
   :exclude-members: start_time, end_time, experiment_id, dataset_example_id, repetition_number, output, error, id, trace_id, annotator_kind, experiment_run_id, from_dict, name, result
