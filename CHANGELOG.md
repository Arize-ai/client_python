# Changelog

## [7.22.5](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.22.4...arize-python-sdk/v7.22.5) (2024-10-08)


### 🐛 Bug Fixes

* Improve export of arize tracing data to improve ergonomics of working with data after export ([#33762](https://github.com/Arize-ai/arize/issues/33762)) ([121678d](https://github.com/Arize-ai/arize/commit/121678db9eebca7d775b9665e60998ff2aaec4a6))

## [7.22.4](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.22.3...arize-python-sdk/v7.22.4) (2024-10-02)


### 📚 Documentation

* Update README.md ([#36223](https://github.com/Arize-ai/arize/issues/36223)) ([10195fe](https://github.com/Arize-ai/arize/commit/10195fe23d0448fc54525fc3359688e751384c5f))

## [7.22.3](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.22.2...arize-python-sdk/v7.22.3) (2024-10-02)


### 📚 Documentation

* update readme ([#36212](https://github.com/Arize-ai/arize/issues/36212)) ([37d35e9](https://github.com/Arize-ai/arize/commit/37d35e9bb33d66182c97992b54fe0860eb9ece39))

## [7.22.2](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.22.1...arize-python-sdk/v7.22.2) (2024-10-02)


### 📚 Documentation

* minot update README.md ([#36207](https://github.com/Arize-ai/arize/issues/36207)) ([9ae117a](https://github.com/Arize-ai/arize/commit/9ae117a0327dc4e589266dbae9c69cdf2024b81d))

## [7.22.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.22.0...arize-python-sdk/v7.22.1) (2024-10-02)


### 📚 Documentation

* Update README.md ([#36173](https://github.com/Arize-ai/arize/issues/36173)) ([f8c73f6](https://github.com/Arize-ai/arize/commit/f8c73f6ae7df0bf2a6bc67e8d673ab8d5ba5b2a0))

## [7.22.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.21.0...arize-python-sdk/v7.22.0) (2024-09-26)


### 🎁 New Features

* **experiments:** configurable concurrency value for task and evaluator runs ([7bfe7ab](https://github.com/Arize-ai/arize/commit/7bfe7abaa673a7567cc20239ebfc7a437d064fd3))
* **experiments:** configurable otlp endpoint for default experiments traces ([7bfe7ab](https://github.com/Arize-ai/arize/commit/7bfe7abaa673a7567cc20239ebfc7a437d064fd3))
* **experiments:** datasets client experiment dry-run mode ([#35678](https://github.com/Arize-ai/arize/issues/35678)) ([7bfe7ab](https://github.com/Arize-ai/arize/commit/7bfe7abaa673a7567cc20239ebfc7a437d064fd3))

### ❔ Miscellaneous Chores

* Move from setup.cfg to pyproject.toml ([#35423](https://github.com/Arize-ai/arize/issues/35423)) ([ff10ea3](https://github.com/Arize-ai/arize/commit/ff10ea3103025d712e693fac808a0645876220ec))

## [7.21.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.20.1...arize-python-sdk/v7.21.0) (2024-09-13)


### 🎁 New Features

* async run task and evaluator functions ([23f5a21](https://github.com/Arize-ai/arize/commit/23f5a2147eb5c76c9620e592972b3fbadfa11057))
* get experiment data back as a dataframe. ([23f5a21](https://github.com/Arize-ai/arize/commit/23f5a2147eb5c76c9620e592972b3fbadfa11057))
* run experiment with default traces ([#34566](https://github.com/Arize-ai/arize/issues/34566)) ([23f5a21](https://github.com/Arize-ai/arize/commit/23f5a2147eb5c76c9620e592972b3fbadfa11057))

### ❔ Miscellaneous Chores

* change max_chunksize to 2**14 ([#34653](https://github.com/Arize-ai/arize/issues/34653)) ([4cf7c86](https://github.com/Arize-ai/arize/commit/4cf7c862726f1f1c1367b712a9ba3b652c6522e9))

## [7.20.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.20.0...arize-python-sdk/v7.20.1) (2024-08-19)


### 🐛 Bug Fixes

* Use numpy `nan` (not `NaN`) for v2 compatibility ([#34274](https://github.com/Arize-ai/arize/issues/34274)) ([c691c31](https://github.com/Arize-ai/arize/commit/c691c31614514cb3d211f9d5aacc28f77cecb579)), closes [#33527](https://github.com/Arize-ai/arize/issues/33527)


### 📚 Documentation

* Add to CHANGELOG for all version 7 ([#34275](https://github.com/Arize-ai/arize/issues/34275)) ([7ac8651](https://github.com/Arize-ai/arize/commit/7ac865135cfea2dd73f9986da4dd3efe1f382ceb)), closes [#34273](https://github.com/Arize-ai/arize/issues/34273)

## [7.20.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.19.0...arize-python-sdk/v7.20.0) (2024-08-16)


### 🎁 New Features

* Enable delayed tags for stream logging ([#34140](https://github.com/Arize-ai/arize/issues/34140)) ([5593127](https://github.com/Arize-ai/arize/commit/559312793b0bff5b21070b87c75d25be30ae2d28))
* Experiment eval metadata ([#34123](https://github.com/Arize-ai/arize/issues/34123)) ([127a9c4](https://github.com/Arize-ai/arize/commit/127a9c42a8f6ed943d2982d924587e92aa4afd55))
* ingest data to arize using space_id ([#33982](https://github.com/Arize-ai/arize/issues/33982)) ([4f349d4](https://github.com/Arize-ai/arize/commit/4f349d47fce675520c54b9d6865cdaa39da7cc54))

## [7.19.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.18.1...arize-python-sdk/v7.19.0) (2024-08-07)


### 🎁 New Features

* Add client to provide public APIs to perform CRUD operations on datasets ([#32096](https://github.com/Arize-ai/arize/issues/32096)) ([512070b](https://github.com/Arize-ai/arize/commit/512070bebad36f11c9e04cb2ab834be125382a6a))
* Allow dataset client to create experiments on datasets ([512070b](https://github.com/Arize-ai/arize/commit/512070bebad36f11c9e04cb2ab834be125382a6a))


## [7.18.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.18.0...arize-python-sdk/v7.18.1) (2024-05-22)


### 🐛 Bug Fixes

* Correctly insert default prediction id column using `df.insert()` ([f938d29](https://github.com/Arize-ai/arize/commit/f938d294581d5a2ff07459fdf77708f1358f9b2d))
* Improve error message for type errors for raw data character count ([#31272](https://github.com/Arize-ai/arize/issues/31272)) ([f938d29](https://github.com/Arize-ai/arize/commit/f938d294581d5a2ff07459fdf77708f1358f9b2d))
* Include SHAP value invalid rows if full of nulls in error message ([#31264](https://github.com/Arize-ai/arize/issues/31264)) ([24f079f](https://github.com/Arize-ai/arize/commit/24f079f3613447e43f08efe11238d7068052dcca))
* Remove f-strings from docstrings and place them above arguments ([#31117](https://github.com/Arize-ai/arize/issues/31117)) ([5c812d6](https://github.com/Arize-ai/arize/commit/5c812d67df6599ac4478135ceb0a60ad23832e52))
* Update similarity timestamp validation ([#31125](https://github.com/Arize-ai/arize/issues/31125)) ([1c32c37](https://github.com/Arize-ai/arize/commit/1c32c37456ff960ba1073a042c300751f80d816a))

## [7.18.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.17.1...arize-python-sdk/v7.18.0) (2024-05-15)


### 🎁 New Features

* python exporter embedding similarity search support ([#30921](https://github.com/Arize-ai/arize/issues/30921)) ([3a58f8f](https://github.com/Arize-ai/arize/commit/3a58f8f4498cd70dac0fe61eb1ef2bc64cb4b057))


### 💫 Code Refactoring

* add preprocessing step for similarity search param in flightserver ([#30984](https://github.com/Arize-ai/arize/issues/30984)) ([34e2a96](https://github.com/Arize-ai/arize/commit/34e2a96b85ed87c107d4327c5d5fd671c3397a1f))

## [7.17.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.17.0...arize-python-sdk/v7.17.1) (2024-05-10)


### 🐛 Bug Fixes

* Serialization of nested dictionaries ([#30931](https://github.com/Arize-ai/arize/issues/30931)) ([ece89dc](https://github.com/Arize-ai/arize/commit/ece89dcf572cbbd1a78eea78e7d4aac13b083c75))
* Avoid side effects in dictionary fields ([ece89dc](https://github.com/Arize-ai/arize/commit/ece89dcf572cbbd1a78eea78e7d4aac13b083c75))
* Value validation should require dictionaries, not JSON ([ece89dc](https://github.com/Arize-ai/arize/commit/ece89dcf572cbbd1a78eea78e7d4aac13b083c75))

## [7.17.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.16.1...arize-python-sdk/v7.17.0) (2024-05-07)


### 🎁 New Features

* Add session and user ids to spans batch logging ([#30588](https://github.com/Arize-ai/arize/issues/30588)) ([90a0416](https://github.com/Arize-ai/arize/commit/90a0416b8278d7bde3d1636ae2a62566956eecfe))
* Send arize schema as part of the request body ([#30841](https://github.com/Arize-ai/arize/issues/30841)) ([b2f8e67](https://github.com/Arize-ai/arize/commit/b2f8e67c488f419dd95ccbcb2cfc3857fd7d9991))


### 🐛 Bug Fixes

* improve evaluation column naming error message ([267d23d](https://github.com/Arize-ai/arize/commit/267d23d949ad675c105099003861146a9b792a4a))
* relax opentelemetry-semantic-conventions dependency ([#30840](https://github.com/Arize-ai/arize/issues/30840)) ([267d23d](https://github.com/Arize-ai/arize/commit/267d23d949ad675c105099003861146a9b792a4a))
* update URL to model page in logger ([#30591](https://github.com/Arize-ai/arize/issues/30591)) ([5f0ee5a](https://github.com/Arize-ai/arize/commit/5f0ee5a3ccdb689b5e5098bacf439a58b5700b4c))

## [7.16.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.16.0...arize-python-sdk/v7.16.1) (2024-04-29)


### 🐛 Bug Fixes

* Add missing `__init__.py` file to tracing validation module ([#30539](https://github.com/Arize-ai/arize/issues/30539)) ([d18f108](https://github.com/Arize-ai/arize/commit/d18f108ebb37b2f4a6add593551703505e472f75))

## [7.16.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.15.0...arize-python-sdk/v7.16.0) (2024-04-24)


### 🎁 New Features

* Add `log_evaluations` method for delayed evaluation logging ([#30179](https://github.com/Arize-ai/arize/issues/30179)) ([0f52763](https://github.com/Arize-ai/arize/commit/0f527630b25686021340fd66680bccf24299f811))
* Fileimporter evaluations updates records ([#30301](https://github.com/Arize-ai/arize/issues/30301)) ([7e1cbf6](https://github.com/Arize-ai/arize/commit/7e1cbf66c6ac11d60bcd56b3319c77c5da8ff448))


### 📚 Documentation

* Add docstring to `log_evaluations` ([0f52763](https://github.com/Arize-ai/arize/commit/0f527630b25686021340fd66680bccf24299f811))
* Add docstring to `log_spans` ([0f52763](https://github.com/Arize-ai/arize/commit/0f527630b25686021340fd66680bccf24299f811))


### 💫 Code Refactoring

* Split `spans` and `evals` validation packages ([#30175](https://github.com/Arize-ai/arize/issues/30175)) ([12efa86](https://github.com/Arize-ai/arize/commit/12efa8676f23709ac6476e4221c997b13af959a0))

## [7.15.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.14.1...arize-python-sdk/v7.15.0) (2024-04-17)


### 🎁 New Features

* Increase embedding raw data character limit ([#30134](https://github.com/Arize-ai/arize/issues/30134)) ([d3e229b](https://github.com/Arize-ai/arize/commit/d3e229ba7f78b9c5a9acc3ee403e085406132e5c))

## [7.14.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.14.0...arize-python-sdk/v7.14.1) (2024-04-03)


### 🐛 Bug Fixes

* Allow spaces in eval names ([#29559](https://github.com/Arize-ai/arize/issues/29559)) ([3879502](https://github.com/Arize-ai/arize/commit/387950253828d006936da7653e060a51543d7b29))

## [7.14.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.13.0...arize-python-sdk/v7.14.0) (2024-03-29)


### 🎁 New Features

* Support export of spans from Arize platform ([#29350](https://github.com/Arize-ai/arize/issues/29350)) ([e248248](https://github.com/Arize-ai/arize/commit/e2482489666f68267286f8bfb8efcf917820d720))
* Increase span field validation string length limits ([#29501](https://github.com/Arize-ai/arize/issues/29501))([c32f464](https://github.com/Arize-ai/arize/commit/c32f464f826cfad948e4f5aa0ad3dff84c460cc1))

## [7.13.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.12.1...arize-python-sdk/v7.13.0) (2024-03-28)


### 🎁 New Features

* Allow sending latent tags without actuals ([#29387](https://github.com/Arize-ai/arize/issues/29387)) ([4c91949](https://github.com/Arize-ai/arize/commit/4c91949bcf861ed98b079953d3210b5276119bca))


### 🐛 Bug Fixes

* use pandas items() vs deprecated iteritems() ([#29384](https://github.com/Arize-ai/arize/issues/29384)) ([0eb377e](https://github.com/Arize-ai/arize/commit/0eb377e6174a9a917ef1e4a579e6874128747a68))

## [7.12.1](https://github.com/Arize-ai/arize/compare/pysdk/7.12.0...arize-python-sdk/v7.12.1) (2024-03-26)


### 💫 Code Refactoring

* **sdk:** Move version to version.py ([#29341](https://github.com/Arize-ai/arize/issues/29341)) ([6e69721](https://github.com/Arize-ai/arize/commit/6e697214ad27e59db7841682465968e37ae51efd))

### ❔ Miscellaneous Chores

* **deps:** Allow pillow version 10+ ([#29376](https://github.com/Arize-ai/arize/issues/29376)) ([6c9e6ed](https://github.com/Arize-ai/arize/commit/6c9e6edefc4adb7a59bc65ac8cf2ab04a2164193))

## [7.12.0](https://github.com/Arize-ai/arize/compare/pysdk/7.11.1...pysdk/7.12.0) (2024-03-23)


### 🎁 New Features

* **evals**: Add evals log spans sdk
* Add certificate file reading to sdk client

### 🐛 Bug Fixes

* Avoid side-effects and correct null validation
* **tracing**: Improve log spans from phoenix
* **tracing**: correct missing value check for explanations
* **tracing**: Import tracing modules dynamically

### 🧪 Tests

* Avoid tracing tests for old python

### 🔀 CI

* Add tracing dependencies to release workflow

## [7.11.1](https://github.com/Arize-ai/arize/compare/pysdk/7.11.0...pysdk/7.11.1) (2024-03-05)


### 🐛 Bug Fixes

* Fix `ImportError` when importing Client from arize.api

## [7.11.0](https://github.com/Arize-ai/arize/compare/pysdk/7.10.2...pysdk/7.11.0) (2024-02-23)


### ❗ Dependency Changes

* Add optional extra dependencies if the Arize package is installed as `pip install arize[NLP_Metrics]`:
  * `nltk>=3.0.0, <4`
  * `sacrebleu>=2.3.1, <3`
  * `rouge-score>=0.1.2, <1`
  * `evaluate>=0.3, <1`
  
### 🎁 New Features

* Add optional strict typing in pandas logger Schema
* Add 0ptional strict typing in record-at-a-time logger


## [7.10.2](https://github.com/Arize-ai/arize/compare/pysdk/7.10.1...pysdk/7.10.2) (2024-02-14)


### 🐛 Bug Fixes

* Address backward compatibility issue for batch logging via Pandas for on-prem customers
* Validate that space and API keys are of string type

## [7.10.1](https://github.com/Arize-ai/arize/compare/pysdk/7.10.0...pysdk/7.10.1) (2024-02-6)


### ❗Dependency Changes:

* Add `deprecated` to our `Tracing` extra requirements. The `deprecated` dependency comes from `opentelemetry-semantic-conventions`, which absence produced an `ImportError`

## [7.10.0](https://github.com/Arize-ai/arize/compare/pysdk/7.9.0...pysdk/7.10.0) (2024-02-1)


### ❗Dependency Updates:

* Relax `MimicExplainer` extra requirements: require only `interpret-community[mimic]>=0.22.0,<1`
 
### 🎁 New Features:

* Add batch ingestion via Pandas DataFrames for `MULTICLASS` model type
* New `TRACING` environment. You can now log spans & traces for your LLM applications into Arize using batch ingestion via Pandas DataFrames
* Removed size limitation on the `Schema`. You can now log wider models (more columns in your DataFrame)
* Prediction ID and Ranking Group ID have an increased character limit from 128 to 512


## [7.9.0](https://github.com/Arize-ai/arize/compare/pysdk/7.8.1...pysdk/7.9.0) (2023-12-28)


### 🎁 New Features:

* New `MULTICLASS` model type available for record-at-a-time ingestion

## [7.8.1](https://github.com/Arize-ai/arize/compare/pysdk/7.8.0...pysdk/7.8.1) (2023-12-18)


### 🐛 Bug Fixes:

* Fix missing columns validation feedback to have repeated columns in the message
* Fix `KeyError` when llm_params is not found in the dataframe. Improved feedback to the user was included.

## [7.8.0](https://github.com/Arize-ai/arize/compare/pysdk/7.7.2...pysdk/7.8.0) (2023-12-13)


### ❗ Dependency Changes

* Updated `pandas` requirement. We now accept pandas `2.x`

### 🎁 New Features

* Enable latent actuals for `GENERATIVE_LLM` models
* Enable feedback when files are too large for better user experience and troubleshooting 

## [7.7.2](https://github.com/Arize-ai/arize/compare/pysdk/7.7.1...pysdk/7.7.2) (2023-11-09)


### 🐛 Bug Fixes:

* Default prediction sent as string for `GENERATIVE_LLM` single-record-logger (before it was incorrectly set as an integer, resulting in it being categorized as prediction score instead of prediction label)

## [7.7.1](https://github.com/Arize-ai/arize/compare/pysdk/7.7.0...pysdk/7.7.1) (2023-11-08)


### 🐛 Bug Fixes:

* Only check the value of `prompt/response` if not `None`

## [7.7.0](https://github.com/Arize-ai/arize/compare/pysdk/7.6.1...pysdk/7.7.0) (2023-11-02)


### 🎁 New Features

* Add `CORPUS` support
* Accept strings for prompt and response
* Make prompt and response optional
* Add support for a list of strings features in single-record-logger

### 🐛 Bug Fixes:

* Avoid creating a view of a Pandas dataframe

## [7.6.1](https://github.com/Arize-ai/arize/compare/pysdk/7.6.0...pysdk/7.6.1) (2023-10-24)


### 🐛 Bug Fixes:

* Add validation on embedding raw data for batch and record-at-a-time loggers
* Raise validation string limits for string fields
* Add truncation warnings for long string fields

## [7.6.0](https://github.com/Arize-ai/arize/compare/pysdk/7.5.1...pysdk/7.6.0) (2023-10-12)


### 🎁 New Features

* Add ability to send features with type list[str]
* Add new fields available to send token usage to Arize, both using our pandas batch logger and the single record logger

## [7.5.1](https://github.com/Arize-ai/arize/compare/pysdk/7.5.0...pysdk/7.5.1) (2023-10-05)


### ❗Dependency Changes

* Require `python>=3.6` (as opposed to `python>=3.8`) for our core SDK. Our extras still require `python>=3.8`. 
* Require `pyarrow>=0.15.0` (as opposed to `pyarrow>=5.0.0`)

### 🐛 Bug Fixes:

* Increase time interval validation from 2 years to 5 years

## [7.5.0](https://github.com/Arize-ai/arize/compare/pysdk/7.4.0...pysdk/7.5.0) (2023-09-02)


### 🎁 New Features

* Add prompt templates and LLM config fields to the single log and pandas batch ingestion. These fields are used in the Arize Prompt Template Playground

### 🐛 Bug Fixes:

* Add a validation check that fails if there are more than 30 embedding features sent

## [7.4.0](https://github.com/Arize-ai/arize/compare/pysdk/7.3.0...pysdk/7.4.0) (2023-08-15)


### 🎁 New Features

* Add filtering via the keyword where to the Exporter client

## [7.3.0](https://github.com/Arize-ai/arize/compare/pysdk/7.2.0...pysdk/7.3.0) (2023-08-01)


### 🎁 New Features

* `AutoEmbeddings` support for any model in the HuggingFace Hub, public or private.
* Add `AutoEmbeddings` UseCase for Object Detection
* Add `EmbeddingGenerator.list_default_models()` method

### Bug Fixes

* Computer Vision `AutoEmbeddings` switched from using `FeatureExtractor`(deprecated from HuggingFace) to `ImageProcessor` class

## [7.2.0](https://github.com/Arize-ai/arize/compare/pysdk/7.1.0...pysdk/7.2.0) (2023-07-22)


### 🎁 New Features

* Authenticating Arize Client using environment variables

### 🐛 Bug Fixes

* Fix permission errors for pandas logging on Windows machines
* Fix enforcement of tags into being strings

## [7.1.0](https://github.com/Arize-ai/arize/compare/pysdk/7.0.6...pysdk/7.1.0) (2023-06-26)


### 🎁 New Features

* Add `Generative_LLM` model-type support for single-record logging

## [7.0.6](https://github.com/Arize-ai/arize/compare/pysdk/7.0.5...pysdk/7.0.6) (2023-06-24)


### ❗Dependency Changes

* Removed dependency on interpret for the MimicExplainer

## [7.0.5](https://github.com/Arize-ai/arize/compare/pysdk/7.0.4...pysdk/7.0.5) (2023-06-23)


### ❗ Dependency Changes

* Add missing dependency for Exporter: tqdm>=4.60.0,<5

### 🐛 Bug Fixes

* Update reserved headers
* **exporter**: Fix progress bar in the Exporter client
* **exporter**: Sort exported dataframe by time
* **exporter**: Add validation check to Exporter client that will fail if start_time > end_time
* **exporter**: Return empty response when an export query returns no data instead of an error.
* **exporter**: Fix the Exporter client returning empty columns in the dataframe if there was no data in them
* Fix incorrect parsing of `GENERATIVE_LLM` model prompt & response fields


## [7.0.4](https://github.com/Arize-ai/arize/compare/pysdk/7.0.3...pysdk/7.0.4) (2023-06-13)


### ❗ Dependency Changes

* Relax protobuf requirements from `protobuf~=3.12` to `protobuf>=3.12, <5`

## [7.0.3](https://github.com/Arize-ai/arize/compare/pysdk/7.0.2...pysdk/7.0.3) (2023-06-02)


### 🎁 New Features

* Add new `ExportClient`, you can now export data from Arize using the Python SDK

### 🐛 Bug Fixes

* Allow `REGRESSION` models to use the `MimicExplainer`
* Remove null value validation for `prediction_label` and `actual_label` from single-record logging
* Add model mapping rules validation for `OBJECT_DETECTION` models

## [7.0.2](https://github.com/Arize-ai/arize/compare/pysdk/7.0.1...pysdk/7.0.2) (2023-05-12)


### ❗ Dependency Changes

* Change optional dependency for `MimicExplainer`, raise the version ceiling of `lightgbm` from `3.3.4` to `4`

### 🐛 Bug Fixes

* Improve error messages around prediction ID, prediction labels, and tags
* Fix predictions sent as scores instead of labels for `NUMERIC` model types
* Add a validation check that will fail if the character limit on tags (1000 max) is exceeded
* Add a validation check that will fail if actuals are sent without prediction ID information (for single-record logging). This would result in a delayed record being sent without a prediction ID, which is necessary for the latent join
* Add a validation check that will fail if the Schema, without prediction columns, does not contain a prediction ID column (for pandas logging). This would result in a delayed record being sent without a prediction ID, which is necessary for the latent join
* Add a validation check that will fail if the Schema points to an empty string as a column name
* Add check for invalid index in AutoEmbeddings: DataFrames must have a sorted, continuous index starting at 0
* Remove label requirements & accept null values on SCORE_CATEGORICAL, NUMERIC, and RANKING models
* Allow feature and tag columns to contain null values for pandas logging
* Allow to send delayed actuals for RANKING models, it is no longer enforced the presence of rank and prediction_group_id columns in the Schema. However, if the columns are sent, they must not have nulls, since we cannot construct predictions with either value null 


## [7.0.1](https://github.com/Arize-ai/arize/compare/pysdk/7.0.0...pysdk/7.0.1) (2023-04-25)

### 🐛 Bug Fixes

* Fix `GENERATIVE_LLM` models being sent as `SCORE_CATEGORICAL` models

## [7.0.0](https://github.com/Arize-ai/arize/compare/pysdk/6.1.3...pysdk/7.0.0) (2023-04-13)


### ⚠ BREAKING CHANGES

* Require `Python >= 3.8` for all extra functionality
* Remove `numeric_sequence` support

### ❗ Dependency Changes

* Add optional extra dependencies if the Arize package is installed as pip install arize[LLM_Evaluation]:

  * nltk>=3.0.0, <4
  * sacrebleu>=2.3.1, <3 
  * rouge-score>=0.1.2, <1
  * evaluate>=0.3, <1

### 🎁 New Features

* Add Object Detection model-type support
* Add Generative LLM model-type support for pandas logging
* Add evaluation metrics generation for Generative LLM models
* Make prediction IDs optional
* Add summarization UseCase to AutoEmbeddings
* Add optional, additional custom headers to Client instantiation
* Add a warning message when only actuals are sent
* Add a descriptive error message when embedding features are sent without a vector
* Add warning when prediction label or prediction ID will be defaulted

### 🐛 Bug Fixes

* A bug causing skipped validation checks when the new REGRESSION and CATEGORICAL model types are selected
* Add a validation check that will fail if the character limit on prediction ID (128 max) is exceeded
* Add a validation check that will fail if there are duplicated columns in the dataframe
* Changed time range requirements to -2/+1 (two years in the past, and 1 future year)
