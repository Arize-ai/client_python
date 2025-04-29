# Changelog

## [7.40.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.40.0...arize-python-sdk/v7.40.1) (2025-04-29)


### 💫 Code Refactoring

* **pandas logger Client, ArizePromptClient(experimental):** deprecate developer_key in favor of api_key in pandas logger Client and ArizePromptClient constructors ([#45037](https://github.com/Arize-ai/arize/issues/45037)) ([0ada819](https://github.com/Arize-ai/arize/commit/0ada819d11648768b5551a89ba3fca7667f5484b))

## [7.40.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.39.0...arize-python-sdk/v7.40.0) (2025-04-24)


### 🎁 New Features

* **experimental, datasets:** deprecate developer_key parameter in ArizeDatasetsClient ([#44926](https://github.com/Arize-ai/arize/issues/44926)) ([dc928a1](https://github.com/Arize-ai/arize/commit/dc928a1c210a097fd1f347859ab914004d157b47))

## [7.39.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.38.1...arize-python-sdk/v7.39.0) (2025-04-24)


### 🎁 New Features

* log_annotations ([#44813](https://github.com/Arize-ai/arize/issues/44813)) ([5f83671](https://github.com/Arize-ai/arize/commit/5f83671ca3e36a779d656936a2a66dee196ad6a0))

## [7.38.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.38.0...arize-python-sdk/v7.38.1) (2025-04-14)


### 🐛 Bug Fixes

* Add and mark underscored headers for deprecation ([#44330](https://github.com/Arize-ai/arize/issues/44330)) ([ecb2a55](https://github.com/Arize-ai/arize/commit/ecb2a5582ae49f6138f0ed270f95d17904e39b86))

## [7.38.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.37.0...arize-python-sdk/v7.38.0) (2025-04-07)


### 🎁 New Features

* **CV:** Image segmentation support ([#43700](https://github.com/Arize-ai/arize/issues/43700)) ([413d531](https://github.com/Arize-ai/arize/commit/413d53164d444cd0fe04ac71fbd0d0fe736c5dea))

## [7.37.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.36.1...arize-python-sdk/v7.37.0) (2025-04-05)


### 🎁 New Features

* **experimental, prompt-hub:** Prompt Hub Client ([#42802](https://github.com/Arize-ai/arize/issues/42802)) ([f59d9b6](https://github.com/Arize-ai/arize/commit/f59d9b6c0c8f15425ee88837244249643d38bc3d))

## [7.36.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.36.0...arize-python-sdk/v7.36.1) (2025-04-04)


### 🐛 Bug Fixes

* **experimental, whylabs-vanguard:** add log_dataset_profile env parameter ([#43902](https://github.com/Arize-ai/arize/issues/43902)) ([31b6843](https://github.com/Arize-ai/arize/commit/31b68433aff483b1568d4faeb73ccf8e2129bc45))

## [7.36.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.35.4...arize-python-sdk/v7.36.0) (2025-03-06)


### 🎁 New Features

* **experimental, online_tasks:** Add `extract_nested_data_to_column` for preprocessing ([#42711](https://github.com/Arize-ai/arize/issues/42711)) ([58cb2d9](https://github.com/Arize-ai/arize/commit/58cb2d963e80ecb1c7bc7c2aece4d311284ae035))


### 🐛 Bug Fixes

* **experimental, whylabs-vanguard:** update client to recent version, require graphql_uri ([#42695](https://github.com/Arize-ai/arize/issues/42695)) ([526de3b](https://github.com/Arize-ai/arize/commit/526de3bf15c9540020b562a30ea039dd87eefa93))

## [7.35.4](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.35.3...arize-python-sdk/v7.35.4) (2025-03-04)


### 🐛 Bug Fixes

* **experimental, whylabs:** profile view support ([#42462](https://github.com/Arize-ai/arize/issues/42462)) ([eb2f0c0](https://github.com/Arize-ai/arize/commit/eb2f0c0374de7e1b1fb19097a05911c976137b75))

## [7.35.3](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.35.2...arize-python-sdk/v7.35.3) (2025-03-01)


### 🐛 Bug Fixes

* **experimental, whylabs-vanguard:** add graphql endpoint parameter, default model env for governance ([#42366](https://github.com/Arize-ai/arize/issues/42366)) ([3c32b91](https://github.com/Arize-ai/arize/commit/3c32b91852bced45729c213e11a11711ca74fabb))

## [7.35.2](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.35.1...arize-python-sdk/v7.35.2) (2025-02-27)


### 🐛 Bug Fixes

* **experimental, whylabs:** fix timestamp logic ([#42170](https://github.com/Arize-ai/arize/issues/42170)) ([cc97a37](https://github.com/Arize-ai/arize/commit/cc97a375313154286db67e1ca3625471c78b3e91))

## [7.35.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.35.0...arize-python-sdk/v7.35.1) (2025-02-21)


### 🐛 Bug Fixes

* **experimental, whylabs-vanguard:** fix profile adapter (generator) name ([#41953](https://github.com/Arize-ai/arize/issues/41953)) ([eff847f](https://github.com/Arize-ai/arize/commit/eff847f6d8cc672202b13a05a3b5d254ce718b0b))

## [7.35.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.34.0...arize-python-sdk/v7.35.0) (2025-02-20)


### 🎁 New Features

* **experimental, whylabs, whylabs-vanguard:** add vanguard governance and ingestion clients, add dataset tags, fix placeholder values  ([#41810](https://github.com/Arize-ai/arize/issues/41810)) ([e1c4933](https://github.com/Arize-ai/arize/commit/e1c4933271abd6c4ae3ecf046a3fa524fc9bdfef))


### 🐛 Bug Fixes

* **experimental, whylabs:** fix row count logic ([#41797](https://github.com/Arize-ai/arize/issues/41797)) ([ac00f7c](https://github.com/Arize-ai/arize/commit/ac00f7c523c74d290e95a83cc1d055d57193288d))

## [7.34.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.33.2...arize-python-sdk/v7.34.0) (2025-02-11)


### 🎁 New Features

* **experimental, whylabs:** integration client implementing log_profile, log_dataset ([#41287](https://github.com/Arize-ai/arize/issues/41287)) ([7ba655f](https://github.com/Arize-ai/arize/commit/7ba655f4c09b3c2d724ce4734e4d7f2f7edced04))


### 🧪 Tests

* **casting:** fix test_casting_config assertion checks ([#41481](https://github.com/Arize-ai/arize/issues/41481)) ([841cb72](https://github.com/Arize-ai/arize/commit/841cb7203c616d9f665ae8ccf8968867c2910b1b))

## [7.33.2](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.33.1...arize-python-sdk/v7.33.2) (2025-02-05)


### 🐛 Bug Fixes

* **experimental, whylabs:** Rename adapter class ([#41259](https://github.com/Arize-ai/arize/issues/41259)) ([8fbf179](https://github.com/Arize-ai/arize/commit/8fbf179e670d3d386d6a7b96b08074b32071181c))

## [7.33.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.33.0...arize-python-sdk/v7.33.1) (2025-02-03)


### 🐛 Bug Fixes 

* **(experimental, whylabs)** Remove implementation of Synth Data Generation from profiles ([#41123](https://github.com/Arize-ai/arize/issues/41123)) ([ea083e1](https://github.com/Arize-ai/arize/commit/ea083e11fa653d9a64bb28ebd7d9c203cb2ed606))

## [7.33.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.32.0...arize-python-sdk/v7.33.0) (2025-01-30)


### 🎁 New Features

* **(experimental)**: synthetic data generator for whylabs integration ([#41002](https://github.com/Arize-ai/arize/issues/41002)) ([e03c084](https://github.com/Arize-ai/arize/commit/e03c084439f708492002ba357c23080acbc4716e))


### 📚 Documentation

* rename datasets to datasets and experiments ([#41034](https://github.com/Arize-ai/arize/issues/41034)) ([ed4c6e7](https://github.com/Arize-ai/arize/commit/ed4c6e739813f08b6b8b1ba2004aa805ca2f8cd0))

## [7.32.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.31.1...arize-python-sdk/v7.32.0) (2025-01-23)


### 🎁 New Features

* support filtering columns on export ([#40618](https://github.com/Arize-ai/arize/pull/40618)) ([ce452b3](https://github.com/Arize-ai/arize/commit/ce452b3b1c89f88056654f0d5314503e50dd9230))


### 🐛 Bug Fixes

* boolean type column in dataframe not showing up in Arize platform after ingested as a dataset ([#40700](https://github.com/Arize-ai/arize/issues/40700)) ([f3b9784](https://github.com/Arize-ai/arize/commit/f3b9784e11e88e674e36f16378806601c495b43b)), closes [#40609](https://github.com/Arize-ai/arize/issues/40609)

## [7.31.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.31.0...arize-python-sdk/v7.31.1) (2025-01-08)


### 🐛 Bug Fixes

* strip scheme from host in exporter client argument ([#39916](https://github.com/Arize-ai/arize/issues/39916)) ([c439b1d](https://github.com/Arize-ai/arize/commit/c439b1d2c978685c4ed64da370ec1927087af2f2))

## [7.31.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.30.0...arize-python-sdk/v7.31.0) (2025-01-07)


### 🎁 New Features

* Add `DatasetSchema` proto object ([#39728](https://github.com/Arize-ai/arize/issues/39728)) ([a68c71e](https://github.com/Arize-ai/arize/commit/a68c71e4c94ca69e050cc14a9eb73caa0355b8fc))


### 🐛 Bug Fixes

* Use the correct type for open inference attributes.llm.tools column  ([#39742](https://github.com/Arize-ai/arize/issues/39742)) ([9345362](https://github.com/Arize-ai/arize/commit/9345362b8ecf8e1832e7dcfce4b01b9d9a84fb75))

## [7.30.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.29.4...arize-python-sdk/v7.30.0) (2024-12-17)


### 🎁 New Features

* Add support for Open Inference llm.tools attribute ([#39077](https://github.com/Arize-ai/arize/issues/39077)) ([8737e6d](https://github.com/Arize-ai/arize/commit/8737e6d125fe9db07f81dc0479e85a1cc30caa0f))

## [7.29.4](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.29.3...arize-python-sdk/v7.29.4) (2024-12-13)


### 🐛 Bug Fixes

* handle pd.NA type for surrogate explainability ([#39135](https://github.com/Arize-ai/arize/issues/39135)) ([547dc57](https://github.com/Arize-ai/arize/commit/547dc57d85d6a46b6529619bf4868ba8ec9395b3))

## [7.29.3](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.29.2...arize-python-sdk/v7.29.3) (2024-12-07)


### 🐛 Bug Fixes

* Add obj parameter to isinstance in _datetime_to_ns ([#38824](https://github.com/Arize-ai/arize/issues/38824)) ([d5bf94f](https://github.com/Arize-ai/arize/commit/d5bf94f5ee262c1d166f2e05eae74fc678148f23))
* update string type handling for surrogate explainability ([#38895](https://github.com/Arize-ai/arize/issues/38895)) ([3499fa2](https://github.com/Arize-ai/arize/commit/3499fa26d5403a9ea038433366aac7bf3bd4c6a1))

## [7.29.2](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.29.1...arize-python-sdk/v7.29.2) (2024-12-03)


### 🐛 Bug Fixes

* Fix tool calls function arguments validation for output_messages when logging spans as a dataframe ([#38628](https://github.com/Arize-ai/arize/issues/38628)) ([661322d](https://github.com/Arize-ai/arize/commit/661322d801b02f33e9ae65d50549d105768dc485))

## [7.29.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.29.0...arize-python-sdk/v7.29.1) (2024-11-27)


### 📚 Documentation

* fix missing classes in api-reference ([#38624](https://github.com/Arize-ai/arize/issues/38624)) ([61019a1](https://github.com/Arize-ai/arize/commit/61019a1212e66e8ae7fefd1939af18aa0ee3cbbe))

## [7.29.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.28.0...arize-python-sdk/v7.29.0) (2024-11-26)


### 🎁 New Features

* **datasets:** log experiment dataframe ([#38573](https://github.com/Arize-ai/arize/issues/38573)) ([9e5ef3c](https://github.com/Arize-ai/arize/commit/9e5ef3c224bcf024225100cf8656845793daa6c1))


### 📚 Documentation

* new two column layout ([#38505](https://github.com/Arize-ai/arize/issues/38505)) ([0a73937](https://github.com/Arize-ai/arize/commit/0a73937a638a524f3cdb819d9b25e8091977a9df))

## [7.28.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.27.2...arize-python-sdk/v7.28.0) (2024-11-25)


### 🎁 New Features

* Increase max character length for multiclass class name to 100 ([#38458](https://github.com/Arize-ai/arize/issues/38458)) ([e4fa486](https://github.com/Arize-ai/arize/commit/e4fa48606a35d11f579fee699d28c35a71312973))

## [7.27.2](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.27.1...arize-python-sdk/v7.27.2) (2024-11-22)


### 📚 Documentation

* Update switcher.json for readthedocs ([#38449](https://github.com/Arize-ai/arize/issues/38449)) ([b1bb67f](https://github.com/Arize-ai/arize/commit/b1bb67fee4b2b05cf9f4a1b349abe807047e8a63))

## [7.27.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.27.0...arize-python-sdk/v7.27.1) (2024-11-22)


### 🐛 Bug Fixes

* allow protobuf v5 for readthedocs doc build ([#38445](https://github.com/Arize-ai/arize/issues/38445)) ([9ef6570](https://github.com/Arize-ai/arize/commit/9ef6570f9fdd8d8466e05d9dd0e6a567946ffbf3))

## [7.27.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.26.0...arize-python-sdk/v7.27.0) (2024-11-22)


### 🎁 New Features

* new `log_evaluations_sync` method to public Flight server ([#37834](https://github.com/Arize-ai/arize/issues/37834)) ([86a21ab](https://github.com/Arize-ai/arize/commit/86a21ab405aba90f0568fb29d918902536bf98a4))
* return number of records updated to Flight clients ([#38428](https://github.com/Arize-ai/arize/issues/38428)) ([b92ee44](https://github.com/Arize-ai/arize/commit/b92ee4451c64c0f1ad26dd8c6de5df0a4d2facd0))

### 💫 Code Refactoring

* Dynamic client key verification on method call ([86a21ab](https://github.com/Arize-ai/arize/commit/86a21ab405aba90f0568fb29d918902536bf98a4))

## [7.26.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.25.7...arize-python-sdk/v7.26.0) (2024-11-18)


### 🎁 New Features

* **datasets:** default experiment traces for the same dataset are sent to Arize in 1 traces model/project ([#38010](https://github.com/Arize-ai/arize/issues/38010)) ([c3face4](https://github.com/Arize-ai/arize/commit/c3face43982bbf177aa2e98ae63aec2ad7b920a3))

## [7.25.7](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.25.6...arize-python-sdk/v7.25.7) (2024-11-12)


### ❔ Miscellaneous Chores

* relax `googleapis-common-protos` and `protobuf`  dependencies ([#37823](https://github.com/Arize-ai/arize/issues/37823)) ([d3eb327](https://github.com/Arize-ai/arize/commit/d3eb32715061d83f7e0277b2ba514041641657b2))

## [7.25.6](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.25.5...arize-python-sdk/v7.25.6) (2024-11-12)


### 🐛 Bug Fixes

* limit dataset identifier hash to 8 characters ([#37817](https://github.com/Arize-ai/arize/issues/37817)) ([e16f0f6](https://github.com/Arize-ai/arize/commit/e16f0f60fbcbd0412b09c888e25d278cd660ddce))

## [7.25.5](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.25.4...arize-python-sdk/v7.25.5) (2024-11-05)


### 📚 Documentation

* new api-reference on homepage and new ml methods ([#37587](https://github.com/Arize-ai/arize/issues/37587)) ([0687cbb](https://github.com/Arize-ai/arize/commit/0687cbbbd634620ddc0405ca259f96540b7d6ccc))

## [7.25.4](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.25.3...arize-python-sdk/v7.25.4) (2024-11-02)


### 🔀 Continuous Integration

* Add workflow to ensure switcher versions for read the docs in release PRs ([#37483](https://github.com/Arize-ai/arize/issues/37483)) ([f9d41fe](https://github.com/Arize-ai/arize/commit/f9d41fe27e2d9a65780eb1fb038f251a5f86defa))

## [7.25.3](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.25.2...arize-python-sdk/v7.25.3) (2024-11-02)


### 📚 Documentation

* Add version switcher dropdown to api docs ([#37478](https://github.com/Arize-ai/arize/issues/37478)) ([c07917c](https://github.com/Arize-ai/arize/commit/c07917c11539375cbf9c187b11515d88f855803b))

## [7.25.2](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.25.1...arize-python-sdk/v7.25.2) (2024-11-01)


### 📚 Documentation

* Add switcher to conf.py in api docs ([4f1ae8c](https://github.com/Arize-ai/arize/commit/4f1ae8c655249a63907fd1a76af8b9e1e063f753))
* Improve docstring format for API docs ([#37475](https://github.com/Arize-ai/arize/issues/37475)) ([4f1ae8c](https://github.com/Arize-ai/arize/commit/4f1ae8c655249a63907fd1a76af8b9e1e063f753))

## [7.25.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.25.0...arize-python-sdk/v7.25.1) (2024-11-01)


### 📚 Documentation

* Add versions to switcher.json ([#37473](https://github.com/Arize-ai/arize/issues/37473)) ([2cd78f0](https://github.com/Arize-ai/arize/commit/2cd78f07aca545494ca026ecaae980d901056de4))

## [7.25.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.24.3...arize-python-sdk/v7.25.0) (2024-11-01)


### 🎁 New Features

* Add mps as available device for embedding generator ([#37471](https://github.com/Arize-ai/arize/issues/37471)) ([a93d686](https://github.com/Arize-ai/arize/commit/a93d686196841b941d89dc9612e5d9eab998bd7e))

## [7.24.3](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.24.2...arize-python-sdk/v7.24.3) (2024-11-01)


### ❔ Miscellaneous Chores

* fix path for readthedocs.yaml ([#37463](https://github.com/Arize-ai/arize/issues/37463)) ([92ef330](https://github.com/Arize-ai/arize/commit/92ef330e5e39fc0024e7cee9a3764b86244ee2c8))

## [7.24.2](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.24.1...arize-python-sdk/v7.24.2) (2024-11-01)


### ❔ Miscellaneous Chores

* Exclude api_reference directory from build ([#37457](https://github.com/Arize-ai/arize/issues/37457)) ([3e595c3](https://github.com/Arize-ai/arize/commit/3e595c37701e4c7384435657d5f3081fa9d8477e))
* Move api docs to docs/ directory ([#37460](https://github.com/Arize-ai/arize/issues/37460)) ([d7d0c21](https://github.com/Arize-ai/arize/commit/d7d0c21603776fc5899356198535af8013d1d91f))

## [7.24.1](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.24.0...arize-python-sdk/v7.24.1) (2024-11-01)


### 📚 Documentation

* Add api_reference directory ([#37423](https://github.com/Arize-ai/arize/issues/37423)) ([c0d338f](https://github.com/Arize-ai/arize/commit/c0d338fea876a03f5ee7bac087341cb69276f1f3))


### 🎨 Styles

* Use ruff as linter/formatter ([#37414](https://github.com/Arize-ai/arize/issues/37414)) ([95c9a35](https://github.com/Arize-ai/arize/commit/95c9a35a83654ac8847461bf2cbbb063e0abdcd8))


### ❔ Miscellaneous Chores

* Add `.readthedocs.yaml` ([#37455](https://github.com/Arize-ai/arize/issues/37455)) ([13a7ad4](https://github.com/Arize-ai/arize/commit/13a7ad436d810b88de6f0c6a8edc923841e97113))


### 💫 Code Refactoring

* Move arize files into src directory ([#37421](https://github.com/Arize-ai/arize/issues/37421)) ([86d9610](https://github.com/Arize-ai/arize/commit/86d96101ae097b08abaee04e1c97a8ee0c31593f))

## [7.24.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.23.0...arize-python-sdk/v7.24.0) (2024-10-30)


### 🎁 New Features

* **datasets:** support using dataset_name in addition to dataset_id to update/delete/get_versions on a dataset ([#37298](https://github.com/Arize-ai/arize/issues/37298)) ([345a18b](https://github.com/Arize-ai/arize/commit/345a18b42552e5d83c020dd39b639f9b418a7d22))
* **experiment:** support callable functions as experiment evaluators ([#37085](https://github.com/Arize-ai/arize/issues/37085)) ([bee9278](https://github.com/Arize-ai/arize/commit/bee92786d9b5079ea51f8e1335f23a5d5b471ee1)), closes [#35779](https://github.com/Arize-ai/arize/issues/35779)


### 🐛 Bug Fixes

* **experiment:** update parameter mapping in task and evaluate functions ([bee9278](https://github.com/Arize-ai/arize/commit/bee92786d9b5079ea51f8e1335f23a5d5b471ee1))

## [7.23.0](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.22.6...arize-python-sdk/v7.23.0) (2024-10-24)


### 🎁 New Features

* Adds project_name support to the Tracing SDK, alongside, to replace model_id, which will be deprecated for Tracing  ([#36962](https://github.com/Arize-ai/arize/issues/36962)) ([be3c129](https://github.com/Arize-ai/arize/commit/be3c12900cfc81c8bb983bb139635bc38b5b35f3))
* **experiments:** add exit_on_error flag to run_experiment API ([#36595](https://github.com/Arize-ai/arize/issues/36595)) ([fc860cd](https://github.com/Arize-ai/arize/commit/fc860cd1a92f3205f709824ffcbf4aba69a8eb9f))
* **experiments:** improve task summary formatting ([fc860cd](https://github.com/Arize-ai/arize/commit/fc860cd1a92f3205f709824ffcbf4aba69a8eb9f))


### 🐛 Bug Fixes

* Move NaN check to before copy which fails on NaN values ([#36503](https://github.com/Arize-ai/arize/issues/36503)) ([1f568bd](https://github.com/Arize-ai/arize/commit/1f568bd09b7ce387e0b38ba84a12ed32a026e5e8))


### 💫 Code Refactoring

* **experiments:** replace print with python logging ([fc860cd](https://github.com/Arize-ai/arize/commit/fc860cd1a92f3205f709824ffcbf4aba69a8eb9f))

## [7.22.6](https://github.com/Arize-ai/arize/compare/arize-python-sdk/v7.22.5...arize-python-sdk/v7.22.6) (2024-10-09)


### 🐛 Bug Fixes

* **experiment:** evaluation result out of order ([#36459](https://github.com/Arize-ai/arize/issues/36459)) ([be484f8](https://github.com/Arize-ai/arize/commit/be484f824082cdf87741dab2dffa4755857b973a))

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
