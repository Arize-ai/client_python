"""Client implementation for managing datasets in the Arize platform."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pyarrow as pa

from arize._flight.client import ArizeFlightClient
from arize._generated.api_client import models
from arize.datasets.validation import validate_dataset_df
from arize.exceptions.base import INVALID_ARROW_CONVERSION_MSG
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.cache import cache_resource, load_cached_resource
from arize.utils.openinference_conversion import (
    convert_boolean_columns_to_str,
    convert_datetime_columns_to_int,
    convert_default_columns_to_json_str,
)
from arize.utils.size import get_payload_size_mb

if TYPE_CHECKING:
    # builtins is needed to use builtins.list in type annotations because
    # the class has a list() method that shadows the built-in list type
    import builtins

    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class DatasetsClient:
    """Client for managing datasets including creation, retrieval, and example management.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The datasets client is a thin wrapper around the generated REST API client,
    using the shared generated API client owned by
    :class:`arize.config.SDKConfiguration`.
    """

    def __init__(
        self, *, sdk_config: SDKConfiguration, generated_client: ApiClient
    ) -> None:
        """
        Args:
            sdk_config: Resolved SDK configuration.
            generated_client: Shared generated API client instance.
        """  # noqa: D205, D212
        self._sdk_config = sdk_config

        # Import at runtime so it's still lazy and extras-gated by the parent
        from arize._generated import api_client as gen

        # Use the provided client directly
        self._api = gen.DatasetsApi(generated_client)

    @prerelease_endpoint(key="datasets.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        space_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.DatasetsList200Response:
        """List datasets the user has access to.

        Datasets are returned in descending creation order (most recently created
        first). Dataset versions are not included in this response; use `get()` to
        retrieve a dataset along with its versions.

        Args:
            space_id: Optional space ID to scope results to a single space.
            limit: Maximum number of datasets to return. The server enforces an
                upper bound.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the datasets and pagination information.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/429).
        """
        return self._api.datasets_list(
            space_id=space_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="datasets.create", stage=ReleaseStage.BETA)
    def create(
        self,
        *,
        name: str,
        space_id: str,
        examples: builtins.list[dict[str, object]] | pd.DataFrame,
        force_http: bool = False,
    ) -> models.Dataset:
        """Create a dataset with JSON examples.

        Empty datasets are not allowed.

        Payload notes (server-enforced):
            - `name` must be unique within the given `space_id`.
            - Each example may contain arbitrary user-defined fields.
            - Do not include system-managed fields on create: `id`, `created_at`,
              `updated_at` (requests containing these fields will be rejected).
            - Each example must contain at least one property (i.e. `{}` is invalid).

        Transport selection:
            - If the payload is below the configured REST payload threshold (or
              `force_http=True`), this method uploads via REST.
            - Otherwise, it attempts a more efficient upload path via gRPC + Flight.

        Args:
            name: Dataset name (must be unique within the target space).
            space_id: Space ID to create the dataset in.
            examples: Dataset examples either as:
                - a list of JSON-like dicts, or
                - a :class:`pandas.DataFrame` (will be converted to records for REST).
            force_http: If True, force REST upload even if the payload exceeds the
                configured REST payload threshold.

        Returns:
            The created dataset object as returned by the API.

        Raises:
            TypeError: If `examples` is not a list of dicts or a :class:`pandas.DataFrame`.
            RuntimeError: If the Flight upload path is selected and the Flight request
                fails.
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/429).
        """
        if not isinstance(examples, list | pd.DataFrame):
            raise TypeError(
                "Examples must be a list of dicts or a pandas DataFrame"
            )
        if len(examples) == 0:
            raise ValueError("Cannot create an empty dataset")

        below_threshold = (
            get_payload_size_mb(examples)
            <= self._sdk_config.max_http_payload_size_mb
        )
        if below_threshold or force_http:
            from arize._generated import api_client as gen

            data = (
                examples.to_dict(orient="records")
                if isinstance(examples, pd.DataFrame)
                else examples
            )

            body = gen.DatasetsCreateRequest(
                name=name,
                space_id=space_id,
                # Cast: pandas to_dict returns dict[Hashable, Any] but API requires dict[str, Any]
                examples=cast("list[dict[str, Any]]", data),
            )
            return self._api.datasets_create(datasets_create_request=body)

        # If we have too many examples, try to convert to a dataframe
        # and log via gRPC + flight
        logger.info(
            f"Uploading {len(examples)} examples via REST may be slow. "
            "Trying to convert to DataFrame for more efficient upload via "
            "gRPC + Flight."
        )
        if not isinstance(examples, pd.DataFrame):
            examples = pd.DataFrame(examples)
        return self._create_dataset_via_flight(
            name=name,
            space_id=space_id,
            examples=examples,
        )

    @prerelease_endpoint(key="datasets.get", stage=ReleaseStage.BETA)
    def get(self, *, dataset_id: str) -> models.Dataset:
        """Get a dataset by ID.

        The returned dataset includes its dataset versions (sorted by creation time,
        most recent first). Dataset examples are not included; use `list_examples()`
        to retrieve examples.

        Args:
            dataset_id: Dataset ID to retrieve.

        Returns:
            The dataset object.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        return self._api.datasets_get(dataset_id=dataset_id)

    @prerelease_endpoint(key="datasets.delete", stage=ReleaseStage.BETA)
    def delete(self, *, dataset_id: str) -> None:
        """Delete a dataset by ID.

        This operation is irreversible.

        Args:
            dataset_id: Dataset ID to delete.

        Returns:
            This method returns None on success (common empty 204 response).

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        return self._api.datasets_delete(dataset_id=dataset_id)

    @prerelease_endpoint(key="datasets.list_examples", stage=ReleaseStage.BETA)
    def list_examples(
        self,
        *,
        dataset_id: str,
        dataset_version_id: str = "",
        limit: int = 100,
        all: bool = False,
    ) -> models.DatasetsExamplesList200Response:
        """List examples for a dataset (optionally for a specific version).

        If `dataset_version_id` is not provided (empty string), the server selects
        the latest dataset version.

        Pagination notes:
            - The response includes `pagination` for forward compatibility.
            - Cursor pagination may not be fully implemented by the server yet.
            - If `all=True`, this method retrieves all examples via the Flight path,
              and returns them in a single response with `has_more=False`.

        Args:
            dataset_id: Dataset ID to list examples for.
            dataset_version_id: Dataset version ID. If empty, the latest version is
                selected.
            limit: Maximum number of examples to return when `all=False`. The server
                enforces an upper bound.
            all: If True, fetch all examples (ignores `limit`) via Flight and return a
                single response.

        Returns:
            A response object containing `examples` and `pagination` metadata.

        Raises:
            RuntimeError: If the Flight request fails or returns no response when
                `all=True`.
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response when `all=False` (e.g. 401/403/404/429).
        """
        if not all:
            return self._api.datasets_examples_list(
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                limit=limit,
            )

        dataset = self.get(dataset_id=dataset_id)
        dataset_updated_at = getattr(dataset, "updated_at", None)
        # TODO(Kiko): Space ID should not be needed,
        # should work on server tech debt to remove this
        space_id = dataset.space_id

        dataset_df = None
        # try to load dataset from cache
        if self._sdk_config.enable_caching:
            dataset_df = load_cached_resource(
                cache_dir=self._sdk_config.cache_dir,
                resource="dataset",
                resource_id=dataset_id,
                resource_updated_at=dataset_updated_at,
            )
        if dataset_df is not None:
            return models.DatasetsExamplesList200Response(
                # Cast: Pydantic validates and converts dicts to DatasetExample at runtime
                examples=cast(
                    "list[models.DatasetExample]",
                    dataset_df.to_dict(orient="records"),
                ),
                pagination=models.PaginationMetadata(
                    has_more=False,  # Note that all=True
                ),
            )

        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            try:
                dataset_df = flight_client.get_dataset_examples(
                    space_id=space_id,
                    dataset_id=dataset_id,
                    dataset_version_id=dataset_version_id,
                )
            except Exception as e:
                msg = f"Error during request: {e!s}"
                logger.exception(msg)
                raise RuntimeError(msg) from e
        if dataset_df is None:
            # This should not happen with proper Flight client implementation,
            # but we handle it defensively
            msg = "No response received from flight server during request"
            logger.error(msg)
            raise RuntimeError(msg)

        # cache dataset for future use
        cache_resource(
            cache_dir=self._sdk_config.cache_dir,
            resource="dataset",
            resource_id=dataset_id,
            resource_updated_at=dataset_updated_at,
            resource_data=dataset_df,
        )

        return models.DatasetsExamplesList200Response(
            # Cast: Pydantic validates and converts dicts to DatasetExample at runtime
            examples=cast(
                "list[models.DatasetExample]",
                dataset_df.to_dict(orient="records"),
            ),
            pagination=models.PaginationMetadata(
                has_more=False,  # Note that all=True
            ),
        )

    # TODO(Kiko): Needs flightserver support
    @prerelease_endpoint(
        key="datasets.append_examples", stage=ReleaseStage.BETA
    )
    def append_examples(
        self,
        *,
        dataset_id: str,
        dataset_version_id: str = "",
        examples: builtins.list[dict[str, object]] | pd.DataFrame,
    ) -> models.Dataset:
        """Append new examples to an existing dataset.

        This method adds examples to an existing dataset version. If
        `dataset_version_id` is not provided (empty string), the server appends
        the examples to the latest dataset version.

        The inserted examples are assigned system-generated IDs by the server.

        Payload requirements (server-enforced):
            - Each example may contain arbitrary user-defined fields.
            - Do not include system-managed fields on input: `id`, `created_at`,
              `updated_at` (requests containing these fields will be rejected).
            - Each example must contain at least one property (i.e. empty
              examples are not invalid).

        Args:
            dataset_id: Dataset ID to append examples to.
            dataset_version_id: Optional dataset version ID to append examples to. If empty,
                the latest dataset version is selected.
            examples: Examples to append, provided as either:
                - a list of JSON-like dicts, or
                - a :class:`pandas.DataFrame` (converted to records before upload).

        Returns:
            The updated dataset object. To see the examples, use `list_examples()`.

        Raises:
            AssertionError: If `examples` is not a list of dicts or a :class:`pandas.DataFrame`.
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/429).
        """
        from arize._generated import api_client as gen

        if not isinstance(examples, list | pd.DataFrame):
            raise TypeError(
                "Examples must be a list of dicts or a pandas DataFrame"
            )

        data = (
            examples.to_dict(orient="records")
            if isinstance(examples, pd.DataFrame)
            else examples
        )
        # Cast: pandas to_dict returns dict[Hashable, Any] but API requires dict[str, Any]
        body = gen.DatasetsExamplesInsertRequest(
            examples=cast("list[dict[str, Any]]", data)
        )

        return self._api.datasets_examples_insert(
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            datasets_examples_insert_request=body,
        )

    def _create_dataset_via_flight(
        self,
        name: str,
        space_id: str,
        examples: pd.DataFrame,
    ) -> models.Dataset:
        """Internal method to create a dataset using Flight protocol for large example sets."""
        data = examples.copy()
        # Convert datetime columns to int64 (ms since epoch)
        data = convert_datetime_columns_to_int(data)
        data = convert_boolean_columns_to_str(data)
        data = _set_default_columns_for_dataset(data)
        data = convert_default_columns_to_json_str(data)

        validation_errors = validate_dataset_df(data)
        if validation_errors:
            raise RuntimeError([e.error_message() for e in validation_errors])

        # Convert to Arrow table
        try:
            logger.debug("Converting data to Arrow format")
            pa_table = pa.Table.from_pandas(data, preserve_index=False)
        except pa.ArrowInvalid as e:
            logger.exception(INVALID_ARROW_CONVERSION_MSG)
            raise pa.ArrowInvalid(
                f"Error converting to Arrow format: {e!s}"
            ) from e
        except Exception:
            logger.exception("Unexpected error creating Arrow table")
            raise

        response = None
        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            try:
                response = flight_client.create_dataset(
                    space_id=space_id,
                    dataset_name=name,
                    pa_table=pa_table,
                )
            except Exception as e:
                msg = f"Error during create request: {e!s}"
                logger.exception(msg)
                raise RuntimeError(msg) from e
        if response is None:
            # This should not happen with proper Flight client implementation,
            # but we handle it defensively
            msg = "No response received from flight server during update"
            logger.error(msg)
            raise RuntimeError(msg)
        # The response from flightserver is the dataset ID. To return the dataset
        # object we make a GET query
        return self.get(dataset_id=response)


def _set_default_columns_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Set default values for created_at and updated_at columns if missing or null."""
    current_time = int(time.time() * 1000)
    if "created_at" in df.columns:
        if df["created_at"].isnull().any():
            df["created_at"].fillna(current_time, inplace=True)
    else:
        df["created_at"] = current_time

    if "updated_at" in df.columns:
        if df["updated_at"].isnull().any():
            df["updated_at"].fillna(current_time, inplace=True)
    else:
        df["updated_at"] = current_time

    if "id" in df.columns:
        if df["id"].isnull().any():
            df["id"] = df["id"].apply(
                lambda x: str(uuid.uuid4()) if pd.isnull(x) else x
            )
    else:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    return df
