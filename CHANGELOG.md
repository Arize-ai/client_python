# Changelog

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

## [7.12.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk-v7.12.0...arize-python-sdk/v7.12.1) (2024-03-26)


### 💫 Code Refactoring

* **sdk:** Move version to version.py ([#29341](https://github.com/Arize-ai/arize/issues/29341)) ([6e69721](https://github.com/Arize-ai/arize/commit/6e697214ad27e59db7841682465968e37ae51efd))

### ❔ Miscellaneous Chores

* **deps:** Allow pillow version 10+ ([#29376](https://github.com/Arize-ai/arize/issues/29376)) ([6c9e6ed](https://github.com/Arize-ai/arize/commit/6c9e6edefc4adb7a59bc65ac8cf2ab04a2164193))
