arize-ai
===============================

version number: 0.0.1
author: Arize Dev

Overview
--------

A helper library to interact with Arize AI APIs

Installation / Usage
--------------------

To install use pip:

    $ pip install arize


Or clone the repo:

    $ git clone https://github.com/Arize-ai/client_python.git
    $ python setup.py install
    
Contributing
------------

TBD

Example
-------

```python
from arize.api import AsyncAPI

arize = AsyncAPI(account_id=1234, api_key='API_KEY')

arize.log(
    model_id='sample-model-v0.0.1',
    prediction_id='eERDCasd9797ca34',
    prediction_value=True,
    labels={'label_key':'label_value', 'label_key1': 'label_value1'}
)
