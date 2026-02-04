"""Client implementation for managing ML models in the Arize platform."""

from __future__ import annotations

import copy
import logging
import time
from typing import TYPE_CHECKING, Any, cast

from arize._generated.protocol.rec import public_pb2 as pb2
from arize._lazy import require
from arize.constants.ml import (
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_NUMBER_OF_EMBEDDINGS,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
    MAX_TAG_LENGTH,
    MAX_TAG_LENGTH_TRUNCATION,
    RESERVED_TAG_COLS,
)
from arize.exceptions.base import (
    INVALID_ARROW_CONVERSION_MSG,
    ValidationFailure,
)
from arize.exceptions.models import MissingModelNameError
from arize.exceptions.parameters import (
    InvalidNumberOfEmbeddings,
    InvalidValueType,
)
from arize.exceptions.spaces import MissingSpaceIDError
from arize.logging import get_truncation_warning_message
from arize.ml.bounded_executor import BoundedExecutor
from arize.ml.casting import cast_dictionary, cast_typed_columns
from arize.ml.stream_validation import (
    validate_and_convert_prediction_id,
    validate_label,
)
from arize.ml.types import (
    CATEGORICAL_MODEL_TYPES,
    NUMERIC_MODEL_TYPES,
    ActualLabelTypes,
    BaseSchema,
    CorpusSchema,
    Embedding,
    EmbeddingColumnNames,
    Environments,
    LLMRunMetadata,
    Metrics,
    ModelTypes,
    PredictionIDType,
    PredictionLabelTypes,
    Schema,
    SimilaritySearchParams,
    TypedValue,
    convert_element,
)
from arize.utils.types import is_list_of

if TYPE_CHECKING:
    import concurrent.futures as cf
    from datetime import datetime

    import pandas as pd
    import requests
    from requests_futures.sessions import FuturesSession

    from arize.config import SDKConfiguration


logger = logging.getLogger(__name__)

_STREAM_DEPS = (
    "requests_futures",
    "google.protobuf",
)
_STREAM_EXTRA = "ml-stream"

_BATCH_DEPS = (
    "pandas",
    "google.protobuf",
    "pyarrow",
    "requests",
    "tqdm",
)
_BATCH_EXTRA = "ml-batch"
_MIMIC_DEPS = (
    "interpret_community.mimic",
    "sklearn.preprocessing",
)
_MIMIC_EXTRA = "mimic-explainer"


class MLModelsClient:
    """Client for logging ML model predictions and actuals to Arize.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.
    """

    def __init__(self, *, sdk_config: SDKConfiguration) -> None:
        """
        Args:
            sdk_config: Resolved SDK configuration.
        """  # noqa: D205, D212
        self._sdk_config = sdk_config

        # internal cache for the futures session
        self._session: FuturesSession | None = None

    def log_stream(
        self,
        *,
        space_id: str,
        model_name: str,
        model_type: ModelTypes,
        environment: Environments,
        model_version: str | None = None,
        prediction_id: PredictionIDType | None = None,
        prediction_timestamp: int | None = None,
        prediction_label: PredictionLabelTypes | None = None,
        actual_label: ActualLabelTypes | None = None,
        features: dict[str, str | bool | float | int | list[str] | TypedValue]
        | None = None,
        embedding_features: dict[str, Embedding] | None = None,
        shap_values: dict[str, float] | None = None,
        tags: dict[str, str | bool | float | int | TypedValue] | None = None,
        batch_id: str | None = None,
        prompt: str | Embedding | None = None,
        response: str | Embedding | None = None,
        prompt_template: str | None = None,
        prompt_template_version: str | None = None,
        llm_model_name: str | None = None,
        llm_params: dict[str, str | bool | float | int] | None = None,
        llm_run_metadata: LLMRunMetadata | None = None,
        timeout: float | None = None,
    ) -> cf.Future:
        """Log a single model prediction or actual to Arize asynchronously.

        This method sends a single prediction, actual, or both to Arize for ML monitoring.
        The request is made asynchronously and returns a Future that can be used to check
        the status or retrieve the response.

        Args:
            space_id: The space ID where the model resides.
            model_name: A unique name to identify your model in the Arize platform.
            model_type: The type of model. Supported types: BINARY, MULTI_CLASS, REGRESSION,
                RANKING, OBJECT_DETECTION. Note: GENERATIVE_LLM is not supported; use the
                spans module instead.
            environment: The environment this data belongs to (PRODUCTION, TRAINING, or
                VALIDATION).
            model_version: Optional version identifier for the model.
            prediction_id: Unique identifier for this prediction. If not provided, one
                will be auto-generated for PRODUCTION environment.
            prediction_timestamp: Unix timestamp (seconds) for when the prediction was made.
                If not provided, the current time is used. Must be within 1 year in the
                future and 2 years in the past from the current time.
            prediction_label: The prediction output from your model. Type depends on
                model_type (e.g., string for categorical, float for numeric).
            actual_label: The ground truth label. Type depends on model_type.
            features: Dictionary of feature name to feature value. Values can be str, bool,
                float, int, list[str], or TypedValue.
            embedding_features: Dictionary of embedding feature name to Embedding object.
                Maximum 50 embeddings per record. Object detection models support only 1.
            shap_values: Dictionary of feature name to SHAP value (float) for feature
                importance/explainability.
            tags: Dictionary of metadata tags. Tag names cannot end with "_shap" or be
                reserved names. Values must be under 1000 characters (warning at 100).
            batch_id: Required for VALIDATION environment; identifies the validation batch.
            prompt: For generative models, the prompt text or embedding sent to the model.
            response: For generative models, the response text or embedding from the model.
            prompt_template: Template used to generate the prompt.
            prompt_template_version: Version identifier for the prompt template.
            llm_model_name: Name of the LLM model used (e.g., "gpt-4").
            llm_params: Dictionary of LLM configuration parameters (e.g., temperature,
                max_tokens).
            llm_run_metadata: Metadata about the LLM run including token counts and latency.
            timeout: Maximum time (in seconds) to wait for the request to complete.

        Returns:
            A concurrent.futures.Future object representing the async request. Call
            .result() to block and retrieve the Response object, or check .done() for
            completion status.

        Raises:
            ValueError: If model_type is GENERATIVE_LLM, or if validation environment is
                missing batch_id, or if training/validation environment is missing
                prediction or actual, or if timestamp is out of range, or if no data
                is provided (must have prediction_label, actual_label, tags, or shap_values),
                or if tag names end with "_shap" or exceed length limits.
            MissingSpaceIDError: If space_id is not provided or empty.
            MissingModelNameError: If model_name is not provided or empty.
            InvalidValueType: If features, tags, or other parameters have incorrect types.
            InvalidNumberOfEmbeddings: If more than 50 embedding features are provided.
            KeyError: If tag names include reserved names.

        Notes:
            - Timestamps must be within 1 year future and 2 years past from current time
            - Tag values are truncated at 1000 characters, with warnings at 100 characters
            - For GENERATIVE_LLM models, use the spans module or OTEL tracing instead
            - The Future returned can be monitored for request status asynchronously
        """
        require(_STREAM_EXTRA, _STREAM_DEPS)
        from arize._generated.protocol.rec import public_pb2 as pb2
        from arize.ml.proto import (
            get_pb_dictionary,
            get_pb_label,
            get_pb_timestamp,
        )

        if model_type == ModelTypes.GENERATIVE_LLM:
            raise ValueError(
                "Wrong model type found: GENERATIVE_LLM. To send LLM data to Arize, "
                "use the spans module `arize_client.spans` or OTEL tracing"
            )

        # This method requires a space_id and project_name
        if not space_id:
            raise MissingSpaceIDError()
        if not model_name:
            raise MissingModelNameError()

        # Validate batch_id
        if environment == Environments.VALIDATION and (
            batch_id is None
            or not isinstance(batch_id, str)
            or len(batch_id.strip()) == 0
        ):
            raise ValueError(
                "Batch ID must be a nonempty string if logging to validation environment."
            )

        # Convert & Validate prediction_id
        prediction_id = validate_and_convert_prediction_id(
            prediction_id,
            environment,
            prediction_label,
            actual_label,
            shap_values,
        )

        # Cast feature & tag values
        if features:
            features = cast_dictionary(features)
            # Defensive check
            if not isinstance(features, dict):
                raise InvalidValueType("features", features, "dict")

            for feat_name, feat_value in features.items():
                _validate_mapping_key(feat_name, "features")
                if is_list_of(feat_value, str):
                    continue
                val = convert_element(feat_value)
                if val is not None and not isinstance(
                    val, (str, bool, float, int)
                ):
                    raise InvalidValueType(
                        f"feature '{feat_name}'",
                        feat_value,
                        "one of: bool, int, float, str",
                    )

        # Validate embedding_features type
        if embedding_features:
            if not isinstance(embedding_features, dict):
                raise InvalidValueType(
                    "embedding_features", embedding_features, "dict"
                )
            if len(embedding_features) > MAX_NUMBER_OF_EMBEDDINGS:
                raise InvalidNumberOfEmbeddings(len(embedding_features))
            if (
                model_type == ModelTypes.OBJECT_DETECTION
                and len(embedding_features.keys()) > 1
            ):
                # Check that there is only 1 embedding feature for OD model types
                raise ValueError(
                    "Object Detection models only support one embedding feature"
                )
            for emb_name, emb_obj in embedding_features.items():
                _validate_mapping_key(emb_name, "embedding features")
                # Must verify embedding type
                if not isinstance(emb_obj, Embedding):
                    raise InvalidValueType(
                        f"embedding feature '{emb_name}'", emb_obj, "Embedding"
                    )
                emb_obj.validate(emb_name)
        if tags:
            tags = cast_dictionary(tags)
            # Defensive check
            if not isinstance(tags, dict):
                raise InvalidValueType("tags", tags, "dict")
            wrong_tags = [
                tag_name for tag_name in tags if tag_name in RESERVED_TAG_COLS
            ]
            if wrong_tags:
                raise KeyError(
                    f"The following tag names are not allowed as they are reserved: {wrong_tags}"
                )
            for tag_name, tag_value in tags.items():
                _validate_mapping_key(tag_name, "tags")
                val = convert_element(tag_value)
                if val is not None and not isinstance(
                    val, (str, bool, float, int)
                ):
                    raise InvalidValueType(
                        f"tag '{tag_name}'",
                        tag_value,
                        "one of: bool, int, float, str",
                    )
                if isinstance(tag_name, str) and tag_name.endswith("_shap"):
                    raise ValueError(
                        f"tag {tag_name} must not be named with a `_shap` suffix"
                    )
                if len(str(val)) > MAX_TAG_LENGTH:
                    raise ValueError(
                        f"The number of characters for each tag must be less than or equal to "
                        f"{MAX_TAG_LENGTH}. The tag {tag_name} with value {tag_value} has "
                        f"{len(str(val))} characters."
                    )
                if len(str(val)) > MAX_TAG_LENGTH_TRUNCATION:
                    logger.warning(
                        get_truncation_warning_message(
                            "tags", MAX_TAG_LENGTH_TRUNCATION
                        )
                    )

        # Check the timestamp present on the event
        if prediction_timestamp is not None:
            if not isinstance(prediction_timestamp, int):
                raise InvalidValueType(
                    "prediction_timestamp", prediction_timestamp, "int"
                )
            # Send warning if prediction is sent with future timestamp
            now = int(time.time())
            if prediction_timestamp > now:
                logger.warning(
                    "Caution when sending a prediction with future timestamp."
                    "Arize only stores 2 years worth of data. For example, if you sent a prediction "
                    "to Arize from 1.5 years ago, and now send a prediction with timestamp of a year in "
                    "the future, the oldest 0.5 years will be dropped to maintain the 2 years worth of data "
                    "requirement."
                )
            if not _is_timestamp_in_range(now, prediction_timestamp):
                raise ValueError(
                    f"prediction_timestamp: {prediction_timestamp} is out of range."
                    f"Prediction timestamps must be within {MAX_FUTURE_YEARS_FROM_CURRENT_TIME} year in the "
                    f"future and {MAX_PAST_YEARS_FROM_CURRENT_TIME} years in the past from the current time."
                )

        # Construct the prediction
        p = None
        if prediction_label is not None:
            if model_version is not None and not isinstance(model_version, str):
                raise InvalidValueType("model_version", model_version, "str")
            validate_label(
                prediction_or_actual="prediction",
                model_type=model_type,
                label=convert_element(prediction_label),
                embedding_features=embedding_features,
            )
            p = pb2.Prediction(
                prediction_label=get_pb_label(
                    prediction_or_actual="prediction",
                    value=prediction_label,
                    model_type=model_type,
                ),
                model_version=model_version,
            )
            if features is not None:
                converted_feats = get_pb_dictionary(features)
                feats = pb2.Prediction(features=converted_feats)
                p.MergeFrom(feats)

            if embedding_features or prompt or response:
                # NOTE: Deep copy is necessary to avoid side effects on the original input dictionary
                combined_embedding_features: dict[str, str | Embedding] = (
                    cast(
                        "dict[str, str | Embedding]", embedding_features.copy()
                    )
                    if embedding_features
                    else {}
                )
                # Map prompt as embedding features for generative models
                if prompt is not None:
                    combined_embedding_features.update({"prompt": prompt})
                # Map response as embedding features for generative models
                if response is not None:
                    combined_embedding_features.update({"response": response})
                converted_embedding_feats = get_pb_dictionary(
                    combined_embedding_features
                )
                embedding_feats = pb2.Prediction(
                    features=converted_embedding_feats
                )
                p.MergeFrom(embedding_feats)

            if tags or llm_run_metadata:
                joined_tags = copy.deepcopy(tags) if tags is not None else {}
                if llm_run_metadata:
                    if llm_run_metadata.total_token_count is not None:
                        joined_tags[
                            LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME
                        ] = llm_run_metadata.total_token_count
                    if llm_run_metadata.prompt_token_count is not None:
                        joined_tags[
                            LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME
                        ] = llm_run_metadata.prompt_token_count
                    if llm_run_metadata.response_token_count is not None:
                        joined_tags[
                            LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME
                        ] = llm_run_metadata.response_token_count
                    if llm_run_metadata.response_latency_ms is not None:
                        joined_tags[
                            LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME
                        ] = llm_run_metadata.response_latency_ms
                converted_tags = get_pb_dictionary(joined_tags)
                tgs = pb2.Prediction(tags=converted_tags)
                p.MergeFrom(tgs)

            if (
                prompt_template
                or prompt_template_version
                or llm_model_name
                or llm_params
            ):
                llm_fields = pb2.LLMFields(
                    prompt_template=prompt_template or "",
                    prompt_template_name=prompt_template_version or "",
                    llm_model_name=llm_model_name or "",
                    llm_params=get_pb_dictionary(llm_params),
                )
                p.MergeFrom(pb2.Prediction(llm_fields=llm_fields))

            if prediction_timestamp is not None:
                p.timestamp.MergeFrom(get_pb_timestamp(prediction_timestamp))

        # Validate and construct the optional actual
        is_latent_tags = prediction_label is None and tags is not None
        a = None
        if actual_label or is_latent_tags:
            a = pb2.Actual()
            if actual_label is not None:
                validate_label(
                    prediction_or_actual="actual",
                    model_type=model_type,
                    label=convert_element(actual_label),
                    embedding_features=embedding_features,
                )
                a.MergeFrom(
                    pb2.Actual(
                        actual_label=get_pb_label(
                            prediction_or_actual="actual",
                            value=actual_label,
                            model_type=model_type,
                        )
                    )
                )
            # Added to support delayed tags on actuals.
            if tags is not None:
                converted_tags = get_pb_dictionary(tags)
                a.MergeFrom(pb2.Actual(tags=converted_tags))

        # Validate and construct the optional feature importances
        fi = None
        if shap_values is not None and bool(shap_values):
            for k, v in shap_values.items():
                if not isinstance(convert_element(v), float):
                    raise InvalidValueType(f"feature '{k}'", v, "float")
                if isinstance(k, str) and k.endswith("_shap"):
                    raise ValueError(
                        f"feature {k} must not be named with a `_shap` suffix"
                    )
            fi = pb2.FeatureImportances(feature_importances=shap_values)

        if p is None and a is None and fi is None:
            raise ValueError(
                "must provide at least one of prediction_label, actual_label, tags, or shap_values"
            )

        env_params = None
        if environment == Environments.TRAINING:
            if p is None or a is None:
                raise ValueError(
                    "Training records must have both Prediction and Actual"
                )
            env_params = pb2.Record.EnvironmentParams(
                training=pb2.Record.EnvironmentParams.Training()
            )
        elif environment == Environments.VALIDATION:
            if p is None or a is None:
                raise ValueError(
                    "Validation records must have both Prediction and Actual"
                )
            env_params = pb2.Record.EnvironmentParams(
                validation=pb2.Record.EnvironmentParams.Validation(
                    batch_id=batch_id
                )
            )
        elif environment == Environments.PRODUCTION:
            env_params = pb2.Record.EnvironmentParams(
                production=pb2.Record.EnvironmentParams.Production()
            )

        rec = pb2.Record(
            # We don't pass the deprecated space key
            # as part of the public record, we pass the space ID in the header
            model_id=model_name,
            prediction_id=prediction_id,
            prediction=p,
            actual=a,
            feature_importances=fi,
            environment_params=env_params,
        )
        headers = self._sdk_config.headers_grpc
        headers.update(
            {
                "Grpc-Metadata-arize-space-id": space_id,
                "Grpc-Metadata-arize-interface": "stream",
            }
        )
        return self._post(
            record=rec,
            headers=headers,
            timeout=timeout,
            indexes=None,  # type: ignore[arg-type]
        )

    def log(
        self,
        *,
        space_id: str,
        model_name: str,
        model_type: ModelTypes,
        dataframe: pd.DataFrame,
        schema: BaseSchema,
        environment: Environments,
        model_version: str = "",
        batch_id: str = "",
        validate: bool = True,
        metrics_validation: list[Metrics] | None = None,
        surrogate_explainability: bool = False,
        timeout: float | None = None,
        tmp_dir: str = "",
    ) -> requests.Response:
        """Log a batch of model predictions and actuals to Arize from a :class:`pandas.DataFrame`.

        This method uploads multiple records to Arize in a single batch operation using
        Apache Arrow format for efficient transfer. The dataframe structure is defined
        by the provided schema which maps dataframe columns to Arize data fields.

        Args:
            space_id: The space ID where the model resides.
            model_name: A unique name to identify your model in the Arize platform.
            model_type: The type of model. Supported types: BINARY, MULTI_CLASS, REGRESSION,
                RANKING, OBJECT_DETECTION. Note: GENERATIVE_LLM is not supported; use the
                spans module instead.
            dataframe (:class:`pandas.DataFrame`): Pandas DataFrame containing the data to
                upload. Columns should correspond to the schema field mappings.
            schema: Schema object (Schema or CorpusSchema) that defines the mapping between
                dataframe columns and Arize data fields (e.g., prediction_label_column_name,
                feature_column_names, etc.).
            environment: The environment this data belongs to (PRODUCTION, TRAINING,
                VALIDATION, or CORPUS).
            model_version: Optional version identifier for the model.
            batch_id: Required for VALIDATION environment; identifies the validation batch.
            validate: When True, performs comprehensive validation before sending data.
                Includes checks for required fields, data types, and value constraints.
            metrics_validation: Optional list of metric families to validate against.
            surrogate_explainability: When True, automatically generates SHAP values using
                MIMIC surrogate explainer. Requires the 'mimic-explainer' extra. Has no
                effect if shap_values_column_names is already specified in schema.
            timeout: Maximum time (in seconds) to wait for the request to complete.
            tmp_dir: Optional temporary directory to store serialized Arrow data before
                upload.

        Returns:
            A requests.Response object from the upload request. Check .status_code for
            success (200) or error conditions.

        Raises:
            MissingSpaceIDError: If space_id is not provided or empty.
            MissingModelNameError: If model_name is not provided or empty.
            ValueError: If model_type is GENERATIVE_LLM, or if environment is CORPUS with
                non-CorpusSchema, or if training/validation records are incomplete.
            ValidationFailure: If validate=True and validation checks fail. Contains list
                of validation error messages.
            pa.ArrowInvalid: If the dataframe cannot be converted to Arrow format, typically
                due to mixed types in columns not specified in the schema.

        Notes:
            - Categorical dtype columns are automatically converted to string
            - Extraneous columns not in the schema are removed before upload
            - Surrogate explainability requires 'mimic-explainer' extra
            - For GENERATIVE_LLM models, use the spans module or OTEL tracing instead
            - If logging actuals without predictions, ensure predictions were logged first
            - Data is sent via Apache Arrow for efficient large batch transfers
        """
        require(_BATCH_EXTRA, _BATCH_DEPS)
        import pandas.api.types as ptypes
        import pyarrow as pa

        from arize.ml.batch_validation.validator import Validator
        from arize.utils.arrow import post_arrow_table
        from arize.utils.dataframe import remove_extraneous_columns

        # This method requires a space_id and project_name
        if not space_id:
            raise MissingSpaceIDError()
        if not model_name:
            raise MissingModelNameError()

        # Deep copy the schema since we might modify it to add certain columns and don't
        # want to cause side effects
        schema = copy.deepcopy(schema)

        if model_type == ModelTypes.GENERATIVE_LLM:
            raise ValueError(
                "Wrong model type found: GENERATIVE_LLM. To send LLM data to Arize, "
                "use the spans module `arize_client.spans` or OTEL tracing"
            )

        # If typed columns are specified in the schema,
        # apply casting and return new copies of the dataframe + schema.
        # All downstream validations are kept the same.
        # note: we don't do any casting for Corpus schemas.
        if isinstance(schema, Schema) and schema.has_typed_columns():
            # The pandas nullable string column type (StringDType) is still considered experimental
            # and is unavailable before pandas 1.0.0.
            # Thus we can only offer this functionality with pandas>=1.0.0.
            try:
                dataframe, schema = cast_typed_columns(dataframe, schema)
            except Exception:
                logger.exception("Error casting typed columns")
                raise

        logger.debug("Performing required validation.")
        errors = Validator.validate_required_checks(
            dataframe=dataframe,
            model_id=model_name,
            environment=environment,
            schema=schema,
            model_version=model_version,
            batch_id=batch_id,
        )
        if errors:
            for e in errors:
                logger.error(e)
            raise ValidationFailure(errors)

        if validate:
            logger.debug("Performing parameters validation.")
            errors = Validator.validate_params(
                dataframe=dataframe,
                model_id=model_name,
                model_type=model_type,
                environment=environment,
                schema=schema,
                metric_families=metrics_validation,
                model_version=model_version,
                batch_id=batch_id,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise ValidationFailure(errors)

        logger.debug("Removing unnecessary columns.")
        dataframe = remove_extraneous_columns(df=dataframe, schema=schema)

        # always validate pd.Category is not present, if yes, convert to string
        # Type ignore: pandas.api.types.is_categorical_dtype exists but stubs may be incomplete
        has_cat_col = any(
            ptypes.is_categorical_dtype(x)  # type: ignore[attr-defined]
            for x in dataframe.dtypes
        )
        if has_cat_col:
            cat_cols = [
                col_name
                for col_name, col_cat in dataframe.dtypes.items()
                if col_cat.name == "category"
            ]
            cat_str_map = dict(
                zip(
                    cat_cols,
                    ["str"] * len(cat_cols),
                    strict=True,
                )
            )
            dataframe = dataframe.astype(cat_str_map)

        if surrogate_explainability:
            require(_MIMIC_EXTRA, _MIMIC_DEPS)
            from arize.ml.surrogate_explainer.mimic import Mimic

            logger.debug("Running surrogate_explainability.")
            # Type ignore: schema typed as BaseSchema but runtime is Schema with these attrs
            if schema.shap_values_column_names:  # type: ignore[attr-defined]
                logger.info(
                    "surrogate_explainability=True has no effect "
                    "because shap_values_column_names is already specified in schema."
                )
            elif schema.feature_column_names is None or (  # type: ignore[attr-defined]
                hasattr(schema.feature_column_names, "__len__")  # type: ignore[attr-defined]
                and len(schema.feature_column_names) == 0  # type: ignore[attr-defined]
            ):
                logger.info(
                    "surrogate_explainability=True has no effect "
                    "because feature_column_names is empty or not specified in schema."
                )
            else:
                dataframe, schema = Mimic.augment(
                    df=dataframe,
                    schema=schema,  # type: ignore[arg-type]
                    model_type=model_type,
                )

        # Convert to Arrow table
        try:
            logger.debug("Converting data to Arrow format")
            # pyarrow will err if a mixed type column exist in the dataset even if
            # the column is not specified in schema. Caveat: There may be other
            # error conditions that we're currently not aware of.
            pa_table = pa.Table.from_pandas(dataframe, preserve_index=False)
        except pa.ArrowInvalid as e:
            logger.exception(INVALID_ARROW_CONVERSION_MSG)
            raise pa.ArrowInvalid(
                f"Error converting to Arrow format: {e!s}"
            ) from e
        except Exception:
            logger.exception("Unexpected error creating Arrow table")
            raise

        if validate:
            logger.debug("Performing types validation.")
            errors = Validator.validate_types(
                model_type=model_type,
                schema=schema,
                pyarrow_schema=pa_table.schema,
            )
            if errors:
                for error in errors:
                    logger.error(error)
                raise ValidationFailure(errors)
        if validate:
            logger.debug("Performing values validation.")
            errors = Validator.validate_values(
                dataframe=dataframe,
                environment=environment,
                schema=schema,
                model_type=model_type,
            )
            if errors:
                for error in errors:
                    logger.error(error)
                raise ValidationFailure(errors)

        if isinstance(schema, Schema) and not schema.has_prediction_columns():
            logger.warning(
                "Logging actuals without any predictions may result in "
                "unexpected behavior if corresponding predictions have not been logged prior. "
                "Please see the docs at https://docs.arize.com/arize/sending-data/sending-data-faq"
                "#what-happens-after-i-send-in-actual-data"
            )

        if environment == Environments.CORPUS:
            proto_schema = _get_pb_schema_corpus(
                schema=schema,  # type: ignore[arg-type]
                model_id=model_name,
            )
        else:
            proto_schema = _get_pb_schema(
                schema=schema,  # type: ignore[arg-type]
                model_id=model_name,
                model_version=model_version,
                model_type=model_type,
                environment=environment,
                batch_id=batch_id,
            )

        # Create headers copy for the spans client
        # Safe to mutate, returns a deep copy
        headers = self._sdk_config.headers
        # Send the number of rows in the dataframe as a header
        # This helps the Arize server to return appropriate feedback, specially for async logging
        headers.update(
            {
                "arize-space-id": space_id,
                "arize-interface": "batch",
                "number-of-rows": str(len(dataframe)),
            }
        )
        return post_arrow_table(
            files_url=self._sdk_config.files_url,
            pa_table=pa_table,
            proto_schema=proto_schema,
            headers=headers,
            timeout=timeout,
            verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
            tmp_dir=tmp_dir,
        )

    def export_to_df(
        self,
        *,
        space_id: str,
        model_name: str,
        environment: Environments,
        start_time: datetime,
        end_time: datetime,
        include_actuals: bool = False,
        model_version: str = "",
        batch_id: str = "",
        where: str = "",
        columns: list | None = None,
        similarity_search_params: SimilaritySearchParams | None = None,
        stream_chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """Export model data from Arize to a :class:`pandas.DataFrame`.

        Retrieves prediction and optional actual data for a model within a specified time
        range and returns it as a :class:`pandas.DataFrame` for analysis.

        Args:
            space_id: The space ID where the model resides.
            model_name: The name of the model to export data from.
            environment: The environment to export from (PRODUCTION, TRAINING, or VALIDATION).
            start_time: Start of the time range (inclusive) as a datetime object.
            end_time: End of the time range (inclusive) as a datetime object.
            include_actuals: When True, includes actual labels in the export. When False,
                only predictions are returned.
            model_version: Optional model version to filter by. Empty string returns all
                versions.
            batch_id: Optional batch ID to filter by (for VALIDATION environment).
            where: Optional SQL-like WHERE clause to filter rows (e.g., "feature_x > 0.5").
            columns: Optional list of column names to include. If None, all columns are
                returned.
            similarity_search_params: Optional parameters for embedding similarity search
                filtering.
            stream_chunk_size: Optional chunk size for streaming large result sets.

        Returns:
            :class:`pandas.DataFrame`: A pandas DataFrame containing the exported data
                with columns for predictions, actuals (if requested), features, tags,
                timestamps, and other model metadata.

        Raises:
            RuntimeError: If the Flight client request fails or returns no response.

        Notes:
            - Uses Apache Arrow Flight for efficient data transfer
            - Large exports may benefit from specifying stream_chunk_size
            - The where clause supports SQL-like filtering syntax
        """
        require(_BATCH_EXTRA, _BATCH_DEPS)
        from arize._exporter.client import ArizeExportClient
        from arize._flight.client import ArizeFlightClient

        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            exporter = ArizeExportClient(
                flight_client=flight_client,
            )
            return exporter.export_to_df(
                space_id=space_id,
                model_id=model_name,
                environment=environment,
                start_time=start_time,
                end_time=end_time,
                where=where,
                columns=columns,
                similarity_search_params=similarity_search_params,
                stream_chunk_size=stream_chunk_size,
                include_actuals=include_actuals,
                model_version=model_version,
                batch_id=batch_id,
            )

    def export_to_parquet(
        self,
        *,
        path: str,
        space_id: str,
        model_name: str,
        environment: Environments,
        start_time: datetime,
        end_time: datetime,
        include_actuals: bool = False,
        model_version: str = "",
        batch_id: str = "",
        where: str = "",
        columns: list | None = None,
        similarity_search_params: SimilaritySearchParams | None = None,
        stream_chunk_size: int | None = None,
    ) -> None:
        """Export model data from Arize to a Parquet file.

        Retrieves prediction and optional actual data for a model within a specified time
        range and writes it directly to a Parquet file at the specified path.

        Args:
            path: The file path where the Parquet file will be written.
            space_id: The space ID where the model resides.
            model_name: The name of the model to export data from.
            environment: The environment to export from (PRODUCTION, TRAINING, or VALIDATION).
            start_time: Start of the time range (inclusive) as a datetime object.
            end_time: End of the time range (inclusive) as a datetime object.
            include_actuals: When True, includes actual labels in the export. When False,
                only predictions are returned.
            model_version: Optional model version to filter by. Empty string returns all
                versions.
            batch_id: Optional batch ID to filter by (for VALIDATION environment).
            where: Optional SQL-like WHERE clause to filter rows (e.g., "feature_x > 0.5").
            columns: Optional list of column names to include. If None, all columns are
                returned.
            similarity_search_params: Optional parameters for embedding similarity search
                filtering.
            stream_chunk_size: Optional chunk size for streaming large result sets.

        Raises:
            RuntimeError: If the Flight client request fails or returns no response.

        Notes:
            - Uses Apache Arrow Flight for efficient data transfer
            - Data is written directly to the specified path as a Parquet file
            - Large exports may benefit from specifying stream_chunk_size
        """
        require(_BATCH_EXTRA, _BATCH_DEPS)
        from arize._exporter.client import ArizeExportClient
        from arize._flight.client import ArizeFlightClient

        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            exporter = ArizeExportClient(
                flight_client=flight_client,
            )
            exporter.export_to_parquet(
                path=path,
                space_id=space_id,
                model_id=model_name,
                environment=environment,
                start_time=start_time,
                end_time=end_time,
                where=where,
                columns=columns,
                similarity_search_params=similarity_search_params,
                stream_chunk_size=stream_chunk_size,
                include_actuals=include_actuals,
                model_version=model_version,
                batch_id=batch_id,
            )

    def _ensure_session(self) -> FuturesSession:
        """Lazily initialize and return the FuturesSession for async streaming requests."""
        from requests_futures.sessions import FuturesSession

        session = object.__getattribute__(self, "_session")
        if session is not None:
            return session

        # disable TLS verification for local dev on localhost, or if user opts out
        new_session = FuturesSession(
            executor=BoundedExecutor(
                self._sdk_config.stream_max_queue_bound,
                self._sdk_config.stream_max_workers,
            )
        )
        object.__setattr__(self, "_session", new_session)
        return new_session

    def _post(
        self,
        record: pb2.Record,
        headers: dict[str, str],
        timeout: float | None,
        indexes: tuple,
    ) -> cf.Future[Any]:
        """Post a record to Arize via async HTTP request with protobuf JSON serialization."""
        from google.protobuf.json_format import MessageToDict

        session = self._ensure_session()
        resp = session.post(
            self._sdk_config.records_url,
            headers=headers,
            timeout=timeout,
            json=MessageToDict(
                message=record,
                preserving_proto_field_name=True,
            ),
            verify=self._sdk_config.request_verify,
        )
        if indexes is not None and len(indexes) == 2:
            resp.starting_index = indexes[0]
            resp.ending_index = indexes[1]
        return resp


def _validate_mapping_key(key_name: str, name: str) -> None:
    """Validate that a mapping key (feature/tag name) is a string and doesn't end with '_shap'."""
    if not isinstance(key_name, str):
        raise TypeError(
            f"{name} dictionary key {key_name} must be named with string, type used: {type(key_name)}"
        )
    if key_name.endswith("_shap"):
        raise ValueError(
            f"{name} dictionary key {key_name} must not be named with a `_shap` suffix"
        )
    return


def _is_timestamp_in_range(now: int, ts: int) -> bool:
    """Check if a timestamp is within the acceptable range (1 year future, 2 years past)."""
    max_time = now + (MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60)
    min_time = now - (MAX_PAST_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60)
    return min_time <= ts <= max_time


def _get_pb_schema(
    schema: Schema,
    model_id: str,
    model_version: str | None,
    model_type: ModelTypes,
    environment: Environments,
    batch_id: str,
) -> object:
    """Construct a protocol buffer Schema from the user's Schema for batch logging."""
    s = pb2.Schema()
    s.constants.model_id = model_id

    if model_version is not None:
        s.constants.model_version = model_version

    if environment == Environments.PRODUCTION:
        s.constants.environment = pb2.Schema.Environment.PRODUCTION
    elif environment == Environments.VALIDATION:
        s.constants.environment = pb2.Schema.Environment.VALIDATION
    elif environment == Environments.TRAINING:
        s.constants.environment = pb2.Schema.Environment.TRAINING
    else:
        raise ValueError(f"unexpected environment: {environment}")

    # Map user-friendly external model types -> internal model types when sending to Arize
    if model_type in NUMERIC_MODEL_TYPES:
        s.constants.model_type = pb2.Schema.ModelType.NUMERIC
    elif model_type in CATEGORICAL_MODEL_TYPES:
        s.constants.model_type = pb2.Schema.ModelType.SCORE_CATEGORICAL
    elif model_type == ModelTypes.RANKING:
        s.constants.model_type = pb2.Schema.ModelType.RANKING
    elif model_type == ModelTypes.OBJECT_DETECTION:
        s.constants.model_type = pb2.Schema.ModelType.OBJECT_DETECTION
    elif model_type == ModelTypes.GENERATIVE_LLM:
        s.constants.model_type = pb2.Schema.ModelType.GENERATIVE_LLM
    elif model_type == ModelTypes.MULTI_CLASS:
        s.constants.model_type = pb2.Schema.ModelType.MULTI_CLASS

    if batch_id is not None:
        s.constants.batch_id = batch_id

    if schema.prediction_id_column_name is not None:
        s.arrow_schema.prediction_id_column_name = (
            schema.prediction_id_column_name
        )

    if schema.timestamp_column_name is not None:
        s.arrow_schema.timestamp_column_name = schema.timestamp_column_name

    if schema.prediction_label_column_name is not None:
        s.arrow_schema.prediction_label_column_name = (
            schema.prediction_label_column_name
        )

    if model_type == ModelTypes.OBJECT_DETECTION:
        if schema.object_detection_prediction_column_names is not None:
            obj_det_pred = schema.object_detection_prediction_column_names
            pred_labels = (
                s.arrow_schema.prediction_object_detection_label_column_names
            )
            pred_labels.bboxes_coordinates_column_name = (
                obj_det_pred.bounding_boxes_coordinates_column_name
            )
            pred_labels.bboxes_categories_column_name = (
                obj_det_pred.categories_column_name
            )
            if obj_det_pred.scores_column_name is not None:
                pred_labels.bboxes_scores_column_name = (
                    obj_det_pred.scores_column_name
                )

        if schema.semantic_segmentation_prediction_column_names is not None:
            seg_pred_cols = schema.semantic_segmentation_prediction_column_names
            pred_seg_labels = s.arrow_schema.prediction_semantic_segmentation_label_column_names
            pred_seg_labels.polygons_coordinates_column_name = (
                seg_pred_cols.polygon_coordinates_column_name
            )
            pred_seg_labels.polygons_categories_column_name = (
                seg_pred_cols.categories_column_name
            )

        if schema.instance_segmentation_prediction_column_names is not None:
            inst_seg_pred_cols = (
                schema.instance_segmentation_prediction_column_names
            )
            pred_inst_seg_labels = s.arrow_schema.prediction_instance_segmentation_label_column_names
            pred_inst_seg_labels.polygons_coordinates_column_name = (
                inst_seg_pred_cols.polygon_coordinates_column_name
            )
            pred_inst_seg_labels.polygons_categories_column_name = (
                inst_seg_pred_cols.categories_column_name
            )
            if inst_seg_pred_cols.scores_column_name is not None:
                pred_inst_seg_labels.polygons_scores_column_name = (
                    inst_seg_pred_cols.scores_column_name
                )
            if (
                inst_seg_pred_cols.bounding_boxes_coordinates_column_name
                is not None
            ):
                pred_inst_seg_labels.bboxes_coordinates_column_name = (
                    inst_seg_pred_cols.bounding_boxes_coordinates_column_name
                )

    if schema.prediction_score_column_name is not None:
        if model_type in NUMERIC_MODEL_TYPES:
            # allow numeric prediction to be sent in as either prediction_label (legacy) or
            # prediction_score.
            s.arrow_schema.prediction_label_column_name = (
                schema.prediction_score_column_name
            )
        else:
            s.arrow_schema.prediction_score_column_name = (
                schema.prediction_score_column_name
            )

    if schema.feature_column_names is not None:
        s.arrow_schema.feature_column_names.extend(schema.feature_column_names)

    if schema.embedding_feature_column_names is not None:
        for (
            emb_name,
            emb_col_names,
        ) in schema.embedding_feature_column_names.items():
            # emb_name is how it will show in the UI
            s.arrow_schema.embedding_feature_column_names_map[
                emb_name
            ].vector_column_name = emb_col_names.vector_column_name
            if emb_col_names.data_column_name:
                s.arrow_schema.embedding_feature_column_names_map[
                    emb_name
                ].data_column_name = emb_col_names.data_column_name
            if emb_col_names.link_to_data_column_name:
                s.arrow_schema.embedding_feature_column_names_map[
                    emb_name
                ].link_to_data_column_name = (
                    emb_col_names.link_to_data_column_name
                )

    if schema.prompt_column_names is not None:
        if isinstance(schema.prompt_column_names, str):
            s.arrow_schema.embedding_feature_column_names_map[
                "prompt"
            ].data_column_name = schema.prompt_column_names
        elif isinstance(schema.prompt_column_names, EmbeddingColumnNames):
            col_names = schema.prompt_column_names
            s.arrow_schema.embedding_feature_column_names_map[
                "prompt"
            ].vector_column_name = col_names.vector_column_name
            if col_names.data_column_name:
                s.arrow_schema.embedding_feature_column_names_map[
                    "prompt"
                ].data_column_name = col_names.data_column_name
    if schema.response_column_names is not None:
        if isinstance(schema.response_column_names, str):
            s.arrow_schema.embedding_feature_column_names_map[
                "response"
            ].data_column_name = schema.response_column_names
        elif isinstance(schema.response_column_names, EmbeddingColumnNames):
            col_names = schema.response_column_names
            s.arrow_schema.embedding_feature_column_names_map[
                "response"
            ].vector_column_name = col_names.vector_column_name
            if col_names.data_column_name:
                s.arrow_schema.embedding_feature_column_names_map[
                    "response"
                ].data_column_name = col_names.data_column_name

    if schema.tag_column_names is not None:
        s.arrow_schema.tag_column_names.extend(schema.tag_column_names)

    if (
        model_type == ModelTypes.RANKING
        and schema.relevance_labels_column_name is not None
    ):
        s.arrow_schema.actual_label_column_name = (
            schema.relevance_labels_column_name
        )
    elif (
        model_type == ModelTypes.RANKING
        and schema.attributions_column_name is not None
    ):
        s.arrow_schema.actual_label_column_name = (
            schema.attributions_column_name
        )
    elif schema.actual_label_column_name is not None:
        s.arrow_schema.actual_label_column_name = (
            schema.actual_label_column_name
        )

    if (
        model_type == ModelTypes.RANKING
        and schema.relevance_score_column_name is not None
    ):
        s.arrow_schema.actual_score_column_name = (
            schema.relevance_score_column_name
        )
    elif schema.actual_score_column_name is not None:
        if model_type in NUMERIC_MODEL_TYPES:
            # allow numeric prediction to be sent in as either prediction_label (legacy) or
            # prediction_score.
            s.arrow_schema.actual_label_column_name = (
                schema.actual_score_column_name
            )
        else:
            s.arrow_schema.actual_score_column_name = (
                schema.actual_score_column_name
            )

    if schema.shap_values_column_names is not None:
        s.arrow_schema.shap_values_column_names.update(
            schema.shap_values_column_names
        )

    if schema.prediction_group_id_column_name is not None:
        s.arrow_schema.prediction_group_id_column_name = (
            schema.prediction_group_id_column_name
        )

    if schema.rank_column_name is not None:
        s.arrow_schema.rank_column_name = schema.rank_column_name

    if model_type == ModelTypes.OBJECT_DETECTION:
        if schema.object_detection_actual_column_names is not None:
            obj_det_actual = schema.object_detection_actual_column_names
            actual_labels = (
                s.arrow_schema.actual_object_detection_label_column_names
            )
            actual_labels.bboxes_coordinates_column_name = (
                obj_det_actual.bounding_boxes_coordinates_column_name
            )
            actual_labels.bboxes_categories_column_name = (
                obj_det_actual.categories_column_name
            )
            if obj_det_actual.scores_column_name is not None:
                actual_labels.bboxes_scores_column_name = (
                    obj_det_actual.scores_column_name
                )

        if schema.semantic_segmentation_actual_column_names is not None:
            sem_seg_actual = schema.semantic_segmentation_actual_column_names
            sem_seg_labels = (
                s.arrow_schema.actual_semantic_segmentation_label_column_names
            )
            sem_seg_labels.polygons_coordinates_column_name = (
                sem_seg_actual.polygon_coordinates_column_name
            )
            sem_seg_labels.polygons_categories_column_name = (
                sem_seg_actual.categories_column_name
            )

        if schema.instance_segmentation_actual_column_names is not None:
            inst_seg_actual = schema.instance_segmentation_actual_column_names
            inst_seg_labels = (
                s.arrow_schema.actual_instance_segmentation_label_column_names
            )
            inst_seg_labels.polygons_coordinates_column_name = (
                inst_seg_actual.polygon_coordinates_column_name
            )
            inst_seg_labels.polygons_categories_column_name = (
                inst_seg_actual.categories_column_name
            )
            if (
                inst_seg_actual.bounding_boxes_coordinates_column_name
                is not None
            ):
                inst_seg_labels.bboxes_coordinates_column_name = (
                    inst_seg_actual.bounding_boxes_coordinates_column_name
                )

    if model_type == ModelTypes.GENERATIVE_LLM:
        if schema.prompt_template_column_names is not None:
            prompt_template_names = schema.prompt_template_column_names
            arrow_prompt_names = s.arrow_schema.prompt_template_column_names
            arrow_prompt_names.template_column_name = (
                prompt_template_names.template_column_name
            )
            arrow_prompt_names.template_version_column_name = (
                prompt_template_names.template_version_column_name
            )
        if schema.llm_config_column_names is not None:
            s.arrow_schema.llm_config_column_names.model_column_name = (
                schema.llm_config_column_names.model_column_name
            )
            s.arrow_schema.llm_config_column_names.params_map_column_name = (
                schema.llm_config_column_names.params_column_name
            )
        if schema.retrieved_document_ids_column_name is not None:
            s.arrow_schema.retrieved_document_ids_column_name = (
                schema.retrieved_document_ids_column_name
            )
    if model_type == ModelTypes.MULTI_CLASS:
        if schema.prediction_score_column_name is not None:
            s.arrow_schema.prediction_score_column_name = (
                schema.prediction_score_column_name
            )
        if schema.multi_class_threshold_scores_column_name is not None:
            s.arrow_schema.multi_class_threshold_scores_column_name = (
                schema.multi_class_threshold_scores_column_name
            )
        if schema.actual_score_column_name is not None:
            s.arrow_schema.actual_score_column_name = (
                schema.actual_score_column_name
            )
    return s


def _get_pb_schema_corpus(
    schema: CorpusSchema,
    model_id: str,
) -> pb2.Schema:
    """Construct a protocol buffer Schema from CorpusSchema for document corpus logging."""
    s = pb2.Schema()
    s.constants.model_id = model_id
    s.constants.environment = pb2.Schema.Environment.CORPUS
    s.constants.model_type = pb2.Schema.ModelType.GENERATIVE_LLM
    if schema.document_id_column_name is not None:
        s.arrow_schema.document_column_names.id_column_name = (
            schema.document_id_column_name
        )
    if schema.document_version_column_name is not None:
        s.arrow_schema.document_column_names.version_column_name = (
            schema.document_version_column_name
        )
    if schema.document_text_embedding_column_names is not None:
        doc_text_emb_cols = schema.document_text_embedding_column_names
        doc_text_col = s.arrow_schema.document_column_names.text_column_name
        doc_text_col.vector_column_name = doc_text_emb_cols.vector_column_name
        doc_text_col.data_column_name = doc_text_emb_cols.data_column_name
        if doc_text_emb_cols.link_to_data_column_name is not None:
            doc_text_col.link_to_data_column_name = (
                doc_text_emb_cols.link_to_data_column_name
            )
    return s
