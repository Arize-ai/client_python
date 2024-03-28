import random
import string
from dataclasses import replace
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from arize.pandas.logger import Schema
from arize.utils.types import CATEGORICAL_MODEL_TYPES, NUMERIC_MODEL_TYPES, ModelTypes
from interpret_community.mimic.mimic_explainer import LGBMExplainableModel, MimicExplainer
from sklearn.preprocessing import LabelEncoder


class Mimic:
    _testing = False

    def __init__(self, X: pd.DataFrame, model_func: Callable):
        self.explainer = MimicExplainer(
            model_func,
            X,
            LGBMExplainableModel,
            augment_data=False,
            is_function=True,
        )

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.explainer.explain_local(X).local_importance_values,
            columns=X.columns,
            index=X.index,
        )

    @staticmethod
    def augment(
        df: pd.DataFrame, schema: Schema, model_type: ModelTypes
    ) -> Tuple[pd.DataFrame, Schema]:
        features = schema.feature_column_names
        X = df[features]

        if X.shape[1] == 0:
            return df, schema

        if model_type in CATEGORICAL_MODEL_TYPES:
            if not schema.prediction_score_column_name:
                raise ValueError(
                    "To calculate surrogate explainability, "
                    f"prediction_score_column_name must be specified in schema for {model_type}."
                )

            y_col_name = schema.prediction_score_column_name
            y = df[y_col_name].to_numpy()

            _min, _max = np.min(y), np.max(y)
            if not 0 <= _min <= 1 or not 0 <= _max <= 1:
                raise ValueError(
                    f"To calculate surrogate explainability for {model_type}, "
                    f"prediction scores must be between 0 and 1, but current "
                    f"prediction scores range from {_min} to {_max}."
                )

            # model func requires 1 positional argument
            def model_func(_):  # type: ignore
                return np.column_stack((1 - y, y))

        elif model_type in NUMERIC_MODEL_TYPES:
            y_col_name = schema.prediction_label_column_name
            if schema.prediction_score_column_name is not None:
                y_col_name = schema.prediction_score_column_name
            y = df[y_col_name].to_numpy()

            _finite_count = np.isfinite(y).sum()
            if len(y) - _finite_count:
                raise ValueError(
                    f"To calculate surrogate explainability for {model_type}, "
                    f"predictions must not contain NaN or infinite values, but "
                    f"{len(y) - _finite_count} NaN or infinite value(s) are found in {y_col_name}."
                )

            # model func requires 1 positional argument
            def model_func(_):  # type: ignore
                return y

        else:
            raise ValueError(
                "Surrogate explainability is not supported for the specified "
                f"model type {model_type}."
            )

        # Column name mapping between features and feature importance values.
        # This is used to augment the schema.
        col_map = {ft: f"{''.join(random.choices(string.ascii_letters, k=8))}" for ft in features}
        aug_schema = replace(schema, shap_values_column_names=col_map)

        # Limit the total number of "cells" to 20M, unless it results in too few or
        # too many rows. This is done to keep the runtime low. Records not sampled
        # have feature importance values set to 0.
        samp_size = min(len(X), min(100_000, max(1_000, 20_000_000 // X.shape[1])))

        if samp_size < len(X):
            _mask = np.zeros(len(X), dtype=int)
            _mask[:samp_size] = 1
            np.random.shuffle(_mask)
            _mask = _mask.astype(bool)
            X = X[_mask]
            y = y[_mask]

        # Apply integer encoding to non-numeric columns.
        # Currently training and explaining detasets are the same, but
        # this can be changed in the future. The student model can be
        # fitted on a much larger dataset since it takes a lot less time.
        X = pd.concat(
            [
                X.select_dtypes(exclude=object),
                pd.DataFrame(
                    {
                        name: LabelEncoder().fit_transform(data)
                        for name, data in X.select_dtypes(object).items()
                    },
                    index=X.index,
                ),
            ],
            axis=1,
        )

        aug_df = pd.concat(
            [
                df,
                Mimic(X, model_func).explain(X).rename(col_map, axis=1),
            ],
            axis=1,
        )

        # Fill null with zero so they're not counted as missing records by server
        if not Mimic._testing:
            aug_df.fillna({c: 0 for c in col_map.values()}, inplace=True)

        return (
            aug_df,
            aug_schema,
        )
