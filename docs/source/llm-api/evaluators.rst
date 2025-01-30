evaluators
^^^^^^^^^^
These are used to create evaluators as a class. `See our docs for more information <https://docs.arize.com/arize/llm-experiments-and-testing/how-to-experiments/create-an-experiment-evaluator>`_. 

To import evaluators, use the following:
``from arize.experimental.datasets.experiments.evaluators.base import ...``

.. autoclass:: arize.experimental.datasets.experiments.evaluators.base.Evaluator
   :members: evaluate, async_evaluate
   :exclude-members: kind, name, _name