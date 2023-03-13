import base64
from itertools import cycle
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.svm import SVC, SVR

from ....arize import public_pb2 as pb2
from ....arize.pandas.logger import Client, Schema
from ....arize.pandas.surrogate_explainer.mimic import Mimic
from ....arize.utils.types import Environments, ModelTypes

Mimic._testing = True


class NoSendClient(Client):
    def __init__(self):
        super().__init__("", "")

    def _post_file(self, path, schema, *_) -> Tuple[pd.DataFrame, Any]:
        s = pb2.Schema()
        s.ParseFromString(base64.b64decode(schema))
        return (
            pa.ipc.open_stream(pa.OSFile(path)).read_pandas(),
            s.arrow_schema,
        )


def _class_df(multiple: int = 1) -> Tuple[pd.DataFrame, Any]:
    # 569 rows, 30 features
    bc = load_breast_cancer()

    features = bc.feature_names
    data, target = bc.data, bc.target
    target_names = bc.target_names

    df = pd.DataFrame(data, columns=features)
    df["_score"] = pd.Series(
        map(lambda v: v[1], SVC(probability=True).fit(df, target).predict_proba(df))
    )
    df["_label"] = df["_score"].apply(lambda s: target_names[int(s > 0.5)])

    # Add a string variable to test integer encoding
    features = np.append(features, "_text")
    df["_text"] = pd.Series(next(cycle("ABCD")) for _ in range(len(df)))

    df["_id"] = pd.Series(range(len(df)))

    if multiple > 1:
        # Make df larger to test sampling
        df = pd.concat([df] * multiple, ignore_index=True)

    # Filtering must not cause problems with indices
    df = df.query("_text == 'A'")

    return df, features


def _reg_df() -> Tuple[pd.DataFrame, Any]:
    # 442 rows, 10 features
    dia = load_diabetes()

    features = dia.feature_names
    data, target = dia.data, dia.target

    df = pd.DataFrame(data, columns=features)
    df["_score"] = pd.Series(SVR().fit(df, target).predict(df))

    # Add a string variable to test integer encoding
    features = np.append(features, "_text")
    df["_text"] = pd.Series(next(cycle("ABCD")) for _ in range(len(df)))

    df["_id"] = pd.Series(range(len(df)))

    # Filtering must not cause problems with indices
    df = df.query("_text == 'A'")

    return df, features


def test_classifier_with_flag_has_shap():
    # Multiply by 1,000 to enlarge df for testing the
    # sampling mechanism. Reduce value to make tests run faster
    df, features = _class_df(1000)

    orig_df = df.copy(deep=True)

    schema = Schema(
        prediction_id_column_name="_id",
        feature_column_names=features,
        prediction_label_column_name="_label",
        prediction_score_column_name="_score",
    )

    try:
        pa_df, pa_schema = NoSendClient().log(
            dataframe=df,
            model_id="classifier",
            model_type=ModelTypes.SCORE_CATEGORICAL,
            environment=Environments.PRODUCTION,
            schema=schema,
            surrogate_explainability=True,
        )
    except Exception:
        assert False

    # must add new columns
    assert pa_df.shape[1] == df.shape[1] + len(features)

    # original portion must not change
    assert df.equals(orig_df)
    assert pa_df[features].equals(df[features])
    assert pa_df[features].equals(orig_df[features])

    # schema must be updated to include feature importance mapping
    k, v = map(set, zip(*pa_schema.shap_values_column_names.items()))
    f = set(features)
    assert set(pa_schema.feature_column_names) == f
    assert k == f  # all features are accounted for
    assert len(v & f) == 0  # no overlap with features
    assert v < set(pa_df.columns)  # new columns exist

    # must have at least one non-zero value
    assert pa_df[pa_schema.shap_values_column_names.values()].count().sum() > 0

    # test the sampling method
    samp_size = min(len(df), min(100_000, max(1_000, 20_000_000 // len(features))))
    _missing = pa_df[pa_schema.shap_values_column_names.values()].isna().sum().sum()
    assert 0 < _missing < len(df) * len(features)
    assert _missing == (len(df) - samp_size) * len(features)


def test_regressor_with_flag_has_shap():
    df, features = _reg_df()

    orig_df = df.copy(deep=True)

    schema = Schema(
        prediction_id_column_name="_id",
        feature_column_names=features,
        prediction_label_column_name="_score",
    )

    try:
        pa_df, pa_schema = NoSendClient().log(
            dataframe=df,
            model_id="regressor",
            model_type=ModelTypes.NUMERIC,
            environment=Environments.PRODUCTION,
            schema=schema,
            surrogate_explainability=True,
        )
    except Exception:
        assert False

    # must add new columns
    assert pa_df.shape[1] == df.shape[1] + len(features)

    # original portion must not change
    assert df.equals(orig_df)
    assert pa_df[features].equals(df[features])
    assert pa_df[features].equals(orig_df[features])

    # schema must be updated to include feature importance mapping
    k, v = map(set, zip(*pa_schema.shap_values_column_names.items()))
    f = set(features)
    assert set(pa_schema.feature_column_names) == f
    assert k == f  # all features are accounted for
    assert len(v & f) == 0  # no overlap with features
    assert v < set(pa_df.columns)  # new columns exist

    # must have at least one non-zero value
    assert pa_df[pa_schema.shap_values_column_names.values()].count().sum() > 0


def test_classifier_no_flag_no_shap():
    df, features = _class_df()

    orig_df = df.copy(deep=True)

    schema = Schema(
        prediction_id_column_name="_id",
        feature_column_names=features,
        prediction_label_column_name="_label",
        prediction_score_column_name="_score",
    )

    try:
        pa_df, pa_schema = NoSendClient().log(
            dataframe=df,
            model_id="classifier",
            model_type=ModelTypes.SCORE_CATEGORICAL,
            environment=Environments.PRODUCTION,
            schema=schema,
        )
    except Exception:
        assert False

    # original dataframe must not change
    assert df.equals(orig_df)
    pa_df = pa_df.sort_index(axis=1)
    assert pa_df.equals(df.sort_index(axis=1))
    assert pa_df.equals(orig_df.sort_index(axis=1))

    # schema must not change
    assert set(pa_schema.feature_column_names) == set(features)
    assert pa_schema.shap_values_column_names == {}


def test_regressor_no_flag_no_shap():
    df, features = _reg_df()
    orig_df = df.copy(deep=True)

    schema = Schema(
        prediction_id_column_name="_id",
        feature_column_names=features,
        prediction_label_column_name="_score",
    )

    try:
        pa_df, pa_schema = NoSendClient().log(
            dataframe=df,
            model_id="regressor",
            model_type=ModelTypes.NUMERIC,
            environment=Environments.PRODUCTION,
            schema=schema,
        )
    except Exception:
        assert False

    # original dataframe must not change
    assert df.equals(orig_df)
    pa_df = pa_df.sort_index(axis=1)
    assert pa_df.equals(df.sort_index(axis=1))
    assert pa_df.equals(orig_df.sort_index(axis=1))

    # schema must not change
    assert set(pa_schema.feature_column_names) == set(features)
    assert pa_schema.shap_values_column_names == {}


def test_classifier_reject_inf_nan():
    for z in (float("inf"), float("-inf"), float("nan")):
        df, features = _class_df()
        df.at[df.index[0], "_score"] = z

        with pytest.raises(ValueError) as e:
            _, _ = NoSendClient().log(
                dataframe=df,
                model_id="classifier",
                model_type=ModelTypes.SCORE_CATEGORICAL,
                environment=Environments.PRODUCTION,
                schema=Schema(
                    prediction_id_column_name="_id",
                    feature_column_names=features,
                    prediction_label_column_name="_label",
                    prediction_score_column_name="_score",
                ),
                surrogate_explainability=True,
            )
        assert "explainability" in str(e.value).lower()


def test_regressor_reject_inf_nan():
    for z in (float("inf"), float("-inf"), float("nan")):
        df, features = _reg_df()
        df.at[df.index[0], "_score"] = z

        with pytest.raises(ValueError) as e:
            _, _ = NoSendClient().log(
                dataframe=df,
                model_id="regressor",
                model_type=ModelTypes.NUMERIC,
                environment=Environments.PRODUCTION,
                schema=Schema(
                    prediction_id_column_name="_id",
                    feature_column_names=features,
                    prediction_label_column_name="_score",
                ),
                surrogate_explainability=True,
            )
        assert "explainability" in str(e.value).lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
