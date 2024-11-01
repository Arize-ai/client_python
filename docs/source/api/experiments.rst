
experiments
===========
Run experiments to test different models, prompts, parameters for your LLM apps. `Read our quickstart guide for more information <https://docs.arize.com/arize/llm-experiments-and-testing/quickstart>`_.

functions
^^^^^^^^^
These are used to run experiments and evaluations. To import functions, use the following:
``from arize.experimental.datasets.experiments.functions import ...``

.. automodule:: arize.experimental.datasets.experiments.functions
   :members: evaluate_experiment, run_experiment
   :exclude-members: get_func_name, get_result_attr, sync_evaluate_run, get_tqdm_progress_bar_formatter

evaluators
^^^^^^^^^^
These are used to create evaluators as a class. `See our docs for more information <https://docs.arize.com/arize/llm-experiments-and-testing/how-to-experiments/create-an-experiment-evaluator>`_. 

To import evaluators, use the following:
``from arize.experimental.datasets.experiments.evaluators.base import ...``

.. autoclass:: arize.experimental.datasets.experiments.evaluators.base.Evaluator
   :members: evaluate, async_evaluate
   :exclude-members: kind, name, _name

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
