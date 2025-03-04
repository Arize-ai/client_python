# type: ignore[pb2]
import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from whylogs.core import DatasetProfile, DatasetProfileView

from arize.experimental.integrations.whylabs.generator import (
    WhylabsProfileAdapter,
)
from arize.pandas.logger import Client
from arize.utils.types import (
    BaseSchema,
    Environments,
    Metrics,
    ModelTypes,
    Schema,
)


class IntegrationClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        space_id: Optional[str] = None,
        space_key: Optional[str] = None,
        uri: Optional[str] = "https://api.arize.com/v1",
        additional_headers: Optional[Dict[str, str]] = None,
        request_verify: Union[bool, str] = True,
        developer_key: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """
        Wrapper for the Arize Client specific to WhyLabs profiles.
        """
        self._client = Client(
            api_key=api_key,
            space_id=space_id,
            space_key=space_key,
            uri=uri,
            additional_headers=additional_headers,
            request_verify=request_verify,
            developer_key=developer_key,
            host=host,
            port=port,
        )
        self._profile_adapter = WhylabsProfileAdapter()

    def log_profile(
        self,
        profile: DatasetProfile | DatasetProfileView,
        schema: BaseSchema,
        environment: Environments,
        model_id: str,
        model_type: ModelTypes,
        metrics_validation: Optional[List[Metrics]] = None,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
        sync: Optional[bool] = False,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
        # Synthetic dataframe generation parameters
        num_rows: Optional[int] = None,
        kll_profile_view: Optional[DatasetProfileView] = None,
        n_kll_quantiles: Optional[int] = 200,
        timestamp: Optional[
            datetime.datetime
        ] = None,  # Overrides timestamp value set in input schema
    ) -> requests.Response:
        """
        Logs a WhyLogs profile to the Arize API.

        The input must be a WhyLogs profile.
        """
        assert isinstance(
            profile, (DatasetProfile, DatasetProfileView)
        ), f"Expected WhyLogs DatasetProfile or DatasetProfileView, got {type(profile)}"

        synthetic_df = self._generate_synthetic_dataset(
            profile,
            num_rows=num_rows or self._extract_num_rows(profile),
            kll_profile_view=kll_profile_view,
            n_kll_quantiles=n_kll_quantiles,
        )

        if timestamp:
            synthetic_df["timestamp"] = timestamp
            # Create a new schema with timestamp added
            schema_dict = schema.asdict()
            schema_dict["timestamp_column_name"] = "timestamp"
            schema = Schema(**schema_dict)

        return self._client.log(
            dataframe=synthetic_df,
            schema=schema,
            environment=environment,
            model_id=model_id,
            model_type=model_type,
            metrics_validation=metrics_validation,
            model_version=model_version,
            batch_id=batch_id,
            sync=sync,
            validate=validate,
            path=path,
            timeout=timeout,
            verbose=verbose,
        )

    def log_dataset_profile(
        self,
        profile: DatasetProfile | DatasetProfileView,
        model_id: str,
        metrics_validation: Optional[List[Metrics]] = None,
        model_version: Optional[str] = None,
        sync: Optional[bool] = False,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
        # Synthetic dataset generation parameters
        num_rows: Optional[int] = None,
        kll_profile_view: Optional[DatasetProfileView] = None,
        n_kll_quantiles: Optional[int] = 200,
        # Schema parameters
        prediction_label_column_name: Optional[str] = None,
        prediction_score_column_name: Optional[str] = None,
        actual_label_column_name: Optional[str] = None,
        actual_score_column_name: Optional[str] = None,
        tag_column_names: Optional[
            List[str]
        ] = None,  # List of columns to be used as tags
        timestamp: Optional[datetime.datetime] = None,
    ) -> requests.Response:
        """
        Logs a WhyLogs dataset-based profile to the Arize API.

        The input must be a WhyLogs profile.
        """
        assert isinstance(
            profile, (DatasetProfile, DatasetProfileView)
        ), f"Expected WhyLogs DatasetProfile or DatasetProfileView, got {type(profile)}"

        synthetic_df = self._generate_synthetic_dataset(
            profile,
            num_rows=num_rows or self._extract_num_rows(profile),
            kll_profile_view=kll_profile_view,
            n_kll_quantiles=n_kll_quantiles,
        )

        schema_args = {
            "feature_column_names": synthetic_df.columns.tolist(),
            "tag_column_names": tag_column_names,
            "prediction_label_column_name": prediction_label_column_name
            or "ARIZE_PLACEHOLDER_STRING",
            "prediction_score_column_name": prediction_score_column_name
            or "ARIZE_PLACEHOLDER_FLOAT",
            "actual_label_column_name": actual_label_column_name
            or "ARIZE_PLACEHOLDER_STRING",
            "actual_score_column_name": actual_score_column_name
            or "ARIZE_PLACEHOLDER_FLOAT",
        }

        # Singular timestamp value for the synthetic dataset
        if timestamp:
            synthetic_df["timestamp"] = timestamp
            schema_args["timestamp_column_name"] = "timestamp"

        schema = Schema(**schema_args)

        synthetic_df["ARIZE_PLACEHOLDER_STRING"] = "ARIZE_PLACEHOLDER"
        synthetic_df["ARIZE_PLACEHOLDER_FLOAT"] = float("-inf")

        return self._client.log(
            dataframe=synthetic_df,
            schema=schema,
            environment=Environments.PRODUCTION,  # Hardcoded for production env
            model_id=model_id,
            model_type=ModelTypes.BINARY_CLASSIFICATION,  # Hardcoded as classification model
            metrics_validation=metrics_validation,
            model_version=model_version,
            sync=sync,
            validate=validate,
            path=path,
            timeout=timeout,
            verbose=verbose,
        )

    def _extract_num_rows(
        self, profile: DatasetProfile | DatasetProfileView
    ) -> int:
        """Extracts number of rows from profile if not explicitly provided."""
        if isinstance(profile, DatasetProfileView):
            profile_df = profile.to_pandas()
        else:
            profile_df = profile.view().to_pandas()

        if "counts/n" not in profile_df.columns:
            raise ValueError("Profile is missing required 'counts/n' column")

        counts = profile_df["counts/n"].unique()
        if len(counts) != 1:
            raise ValueError(
                "Multiple different values detected in 'counts/n' column. "
                "Please specify num_rows parameter explicitly or pass in a "
                "profile with a consistent value for the 'counts/n' column."
            )

        return counts[0]

    def _generate_synthetic_dataset(
        self,
        profile: DatasetProfile | DatasetProfileView,
        num_rows: int,
        kll_profile_view: Optional[DatasetProfileView] = None,
        n_kll_quantiles: Optional[int] = 200,
    ) -> pd.DataFrame:
        if isinstance(profile, DatasetProfileView):
            profile_df = profile.to_pandas()
        else:
            profile_df = profile.view().to_pandas()

        return self._profile_adapter.generate(
            profile_df,
            num_rows=num_rows,
            kll_profile_view=kll_profile_view,
            n_kll_quantiles=n_kll_quantiles,
        )
