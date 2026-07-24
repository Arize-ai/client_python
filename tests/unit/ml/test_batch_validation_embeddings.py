"""Tests for embedding-vector value validation in the ML batch Validator.

Covers ``Validator._check_embedding_vectors_have_valid_values``, which rejects
embedding columns that would be silently dropped during server-side ingestion:
all-empty columns and columns holding empty / non-finite (NaN/Inf) vectors. See
the ``InvalidValueEmbeddingVectorHasNoValues`` error for the two failure modes.

The client-side classification mirrors the server (``validateEmbedding`` ->
``IsNumericVector`` in ``go/pkg/lib/validation/common.go``): a null vector is an
allowed placeholder, but a non-null vector must be non-empty and every element
must be finite — a single NaN/Inf makes the whole vector invalid.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from arize.exceptions.types import (
    InvalidValueEmbeddingVectorDimensionality,
    InvalidValueEmbeddingVectorHasNoValues,
)
from arize.ml.batch_validation.validator import Validator
from arize.ml.types import (
    EmbeddingColumnNames,
    Environments,
    ModelTypes,
    Schema,
)


def _schema(*vector_cols: str) -> Schema:
    return Schema(
        prediction_id_column_name="prediction_id",
        embedding_feature_column_names={
            col: EmbeddingColumnNames(vector_column_name=col)
            for col in vector_cols
        },
    )


@pytest.mark.unit
class TestCheckEmbeddingVectorsHaveValidValues:
    """Value validation for declared embedding vector columns."""

    def test_no_embedding_features_returns_no_errors(self) -> None:
        """Schema without embedding features is a no-op."""
        df = pd.DataFrame({"prediction_id": ["a", "b"]})
        schema = Schema(prediction_id_column_name="prediction_id")

        assert (
            Validator._check_embedding_vectors_have_valid_values(df, schema)
            == []
        )

    def test_all_valid_vectors_pass(self) -> None:
        """Fully populated finite vectors produce no error."""
        df = pd.DataFrame(
            {"good": [np.arange(4.0), np.arange(4.0), np.arange(4.0)]}
        )

        assert (
            Validator._check_embedding_vectors_have_valid_values(
                df, _schema("good")
            )
            == []
        )

    def test_sparse_none_placeholders_pass(self) -> None:
        """A None vector alongside real vectors is an allowed placeholder."""
        df = pd.DataFrame({"good": [np.arange(4.0), None, np.arange(4.0)]})

        assert (
            Validator._check_embedding_vectors_have_valid_values(
                df, _schema("good")
            )
            == []
        )

    def test_all_none_column_flagged_as_empty(self) -> None:
        """A column that is entirely None has no usable vectors."""
        df = pd.DataFrame({"empty": [None, None, None]})

        errors = Validator._check_embedding_vectors_have_valid_values(
            df, _schema("empty")
        )

        assert len(errors) == 1
        assert isinstance(errors[0], InvalidValueEmbeddingVectorHasNoValues)
        assert errors[0].all_empty_cols == ["empty"]
        assert errors[0].invalid_value_cols == []

    def test_all_nan_only_column_flagged_as_empty(self) -> None:
        """A column whose every vector is all-NaN has no usable vectors."""
        df = pd.DataFrame(
            {
                "nan_only": [
                    np.array([np.nan, np.nan]),
                    np.array([None, np.nan], dtype=float),
                ]
            }
        )

        errors = Validator._check_embedding_vectors_have_valid_values(
            df, _schema("nan_only")
        )

        assert len(errors) == 1
        assert errors[0].all_empty_cols == ["nan_only"]
        assert errors[0].invalid_value_cols == []

    def test_mixed_all_nan_and_valid_flagged(self) -> None:
        """A column with some all-NaN vectors but also real ones is reported.

        A batch that mixes all-NaN vectors with valid ones is dropped
        file-wide by the server, so it must be caught client-side.
        """
        df = pd.DataFrame(
            {
                "mixed": [
                    np.array([np.nan, np.nan]),
                    np.arange(3.0),
                    np.array([None, np.nan], dtype=float),
                ]
            }
        )

        errors = Validator._check_embedding_vectors_have_valid_values(
            df, _schema("mixed")
        )

        assert len(errors) == 1
        assert errors[0].all_empty_cols == []
        assert errors[0].invalid_value_cols == ["mixed"]

    def test_partial_nan_vector_flagged(self) -> None:
        """A vector with a single NaN is invalid, matching the server.

        The server's IsNumericVector rejects a vector if ANY element is
        non-finite, so partial-NaN vectors like [1.0, NaN, 3.0] must be caught
        client-side too — otherwise they trigger the same silent file drop.
        """
        df = pd.DataFrame(
            {"partial": [np.array([1.0, np.nan, 3.0]), np.arange(3.0)]}
        )

        errors = Validator._check_embedding_vectors_have_valid_values(
            df, _schema("partial")
        )

        assert len(errors) == 1
        assert errors[0].invalid_value_cols == ["partial"]

    def test_inf_vector_flagged(self) -> None:
        """Inf/-Inf elements are rejected identically to NaN."""
        df = pd.DataFrame(
            {"inf": [np.array([np.inf, -np.inf]), np.arange(3.0)]}
        )

        errors = Validator._check_embedding_vectors_have_valid_values(
            df, _schema("inf")
        )

        assert len(errors) == 1
        assert errors[0].invalid_value_cols == ["inf"]

    def test_empty_list_vector_flagged(self) -> None:
        """An empty-list vector counts as having no values."""
        df = pd.DataFrame({"e": [[], np.arange(3.0)]})

        errors = Validator._check_embedding_vectors_have_valid_values(
            df, _schema("e")
        )

        assert len(errors) == 1
        assert errors[0].invalid_value_cols == ["e"]

    def test_multiple_columns_partitioned_correctly(self) -> None:
        """Empty and invalid-value columns are reported in their own buckets."""
        df = pd.DataFrame(
            {
                "good": [np.arange(4.0), np.arange(4.0), None],
                "all_empty": [None, None, None],
                "mixed": [
                    np.array([np.nan, np.nan]),
                    np.arange(3.0),
                    None,
                ],
            }
        )

        errors = Validator._check_embedding_vectors_have_valid_values(
            df, _schema("good", "all_empty", "mixed")
        )

        assert len(errors) == 1
        assert errors[0].all_empty_cols == ["all_empty"]
        assert errors[0].invalid_value_cols == ["mixed"]

    def test_column_missing_from_dataframe_is_skipped(self) -> None:
        """Declared vector columns absent from the frame are ignored here."""
        df = pd.DataFrame({"present": [np.arange(4.0), np.arange(4.0)]})

        assert (
            Validator._check_embedding_vectors_have_valid_values(
                df, _schema("present", "absent")
            )
            == []
        )

    def test_scalar_null_sentinels_treated_as_placeholders(self) -> None:
        """Every scalar missing sentinel is an allowed null placeholder.

        A column like [float("nan"), np.arange(3)] is inferred by PyArrow as
        list<double>, so the scalar NaN reaches value validation as a bare
        float. It — along with np.float64(NaN), pd.NA and pd.NaT — must be
        classified as a null placeholder (not a zero-length invalid vector),
        matching the server, which lets null rows flow through.
        """
        for sentinel in (
            float("nan"),
            np.float32(np.nan),
            np.float64(np.nan),
            pd.NA,
            pd.NaT,
        ):
            df = pd.DataFrame({"emb": [sentinel, np.arange(3.0)]})

            assert (
                Validator._check_embedding_vectors_have_valid_values(
                    df, _schema("emb")
                )
                == []
            ), sentinel

    def test_scalar_null_sentinel_only_column_flagged_as_empty(self) -> None:
        """A column of only scalar null sentinels has no usable vectors."""
        df = pd.DataFrame({"emb": [pd.NA, float("nan"), None]})

        errors = Validator._check_embedding_vectors_have_valid_values(
            df, _schema("emb")
        )

        assert len(errors) == 1
        assert errors[0].all_empty_cols == ["emb"]
        assert errors[0].invalid_value_cols == []


@pytest.mark.unit
class TestCheckEmbeddingVectorsDimensionality:
    """Dimensionality validation must not choke on scalar null sentinels."""

    def test_scalar_null_sentinels_do_not_raise(self) -> None:
        """Scalar NaN/NA sentinels count as dim 0 instead of raising.

        The dimensionality check runs before the value check, and it used to
        call len() on anything that was not the exact np.nan singleton — so a
        list<double> column carrying a bare float("nan"), np.float64(NaN),
        pd.NA or pd.NaT raised ``TypeError: object of type 'float' has no
        len()`` before the value check could classify the null placeholder.
        """
        for sentinel in (
            float("nan"),
            np.float32(np.nan),
            np.float64(np.nan),
            pd.NA,
            pd.NaT,
        ):
            df = pd.DataFrame({"emb": [sentinel, np.arange(3.0)]})

            # dim-0 placeholder + a valid 3-d vector => no dimensionality error
            assert (
                Validator._check_embedding_vectors_dimensionality(
                    df, _schema("emb")
                )
                == []
            ), sentinel

    def test_scalar_null_sentinel_does_not_mask_low_dim(self) -> None:
        """A genuine dim-1 vector is still reported alongside a null sentinel."""
        df = pd.DataFrame({"emb": [float("nan"), np.array([1.0])]})

        errors = Validator._check_embedding_vectors_dimensionality(
            df, _schema("emb")
        )

        assert len(errors) == 1
        assert isinstance(errors[0], InvalidValueEmbeddingVectorDimensionality)
        assert errors[0].dim_1_cols == ["emb"]


@pytest.mark.unit
class TestValidateTypesAndValuesEmbeddingRegression:
    """Full validate_types + validate_values path for scalar-NaN embeddings.

    Regression for the public pandas upload path: a column mixing a scalar NaN
    with real vectors is inferred as list<double>, passes type validation, and
    must then flow through value validation without raising.
    """

    @staticmethod
    def _run(df: pd.DataFrame, schema: Schema) -> list:
        pa_schema = pa.Table.from_pandas(df, preserve_index=False).schema
        type_errors = Validator.validate_types(
            model_type=ModelTypes.SCORE_CATEGORICAL,
            schema=schema,
            pyarrow_schema=pa_schema,
        )
        assert type_errors == []
        return Validator.validate_values(
            dataframe=df,
            environment=Environments.PRODUCTION,
            schema=schema,
            model_type=ModelTypes.SCORE_CATEGORICAL,
            max_past_years=5,
        )

    def test_scalar_nan_placeholder_column_is_valid(self) -> None:
        """[float("nan"), vector] validates cleanly end-to-end (no TypeError)."""
        df = pd.DataFrame(
            {
                "prediction_id": ["a", "b"],
                "emb": [float("nan"), np.arange(3.0)],
            }
        )

        assert self._run(df, _schema("emb")) == []

    def test_scalar_nan_with_invalid_vector_is_reported(self) -> None:
        """A scalar NaN placeholder does not hide a genuinely invalid vector."""
        df = pd.DataFrame(
            {
                "prediction_id": ["a", "b", "c"],
                "emb": [
                    float("nan"),
                    np.arange(3.0),
                    np.array([1.0, np.nan, 3.0]),
                ],
            }
        )

        errors = self._run(df, _schema("emb"))

        assert len(errors) == 1
        assert isinstance(errors[0], InvalidValueEmbeddingVectorHasNoValues)
        assert errors[0].invalid_value_cols == ["emb"]


@pytest.mark.unit
class TestInvalidValueEmbeddingVectorHasNoValues:
    """The error's message and representation."""

    def test_repr(self) -> None:
        error = InvalidValueEmbeddingVectorHasNoValues([], [])
        assert repr(error) == "Invalid_Value_Embedding_Vector_Has_No_Values"

    def test_message_lists_both_buckets(self) -> None:
        error = InvalidValueEmbeddingVectorHasNoValues(
            all_empty_cols=["col_a"], invalid_value_cols=["col_b"]
        )
        message = error.error_message()

        assert "col_a" in message
        assert "col_b" in message
        assert "finite numeric values" in message

    def test_message_omits_empty_bucket(self) -> None:
        error = InvalidValueEmbeddingVectorHasNoValues(
            all_empty_cols=["col_a"], invalid_value_cols=[]
        )
        message = error.error_message()

        assert "col_a" in message
        assert "non-null vectors that" not in message
