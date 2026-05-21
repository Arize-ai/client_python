"""Arize SDK for model observability and LLM tracing."""

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, cast

from arize._generated.api_client import models
from arize._generated.api_client.exceptions import ApiException
from arize._generated.api_client.models.problem import Problem
from arize.annotation_configs.types import (
    AnnotationConfigListResponse as _DomainAnnotationConfigListResponse,
)
from arize.client import ArizeClient
from arize.config import SDKConfiguration
from arize.evaluators.types import (
    EvaluatorVersionListResponse as _DomainEvaluatorVersionListResponse,
)
from arize.exceptions.spaces import AmbiguousNameError
from arize.regions import Region
from arize.tasks.types import (
    TaskListResponse as _DomainTaskListResponse,
)
from arize.users.types import BulkUserDeletionResult
from arize.users.types import (
    UserListResponse as _DomainUserListResponse,
)
from arize.utils.resolve import NotFoundError

if TYPE_CHECKING:
    import pandas as pd

# Attach a NullHandler by default in the top-level package
# so that if no configuration is installed, nothing explodes.
logging.getLogger("arize").addHandler(logging.NullHandler())

# Opt-in env-based logging
try:
    from .logging import auto_configure_from_env

    auto_configure_from_env()
except Exception:  # noqa: S110
    # Intentionally silent: logging configuration is optional and should never
    # prevent SDK initialization. Users can configure logging explicitly if needed.
    pass

__all__ = [
    "AmbiguousNameError",
    "ApiException",
    "ArizeClient",
    "BulkUserDeletionResult",
    "NotFoundError",
    "Problem",
    "Region",
    "SDKConfiguration",
]


def _pivot_annotations(row: dict, prefix: str = "annotation") -> dict:
    """Replace the ``annotations`` list with per-annotation named columns.

    Each ``Annotation`` dict is expanded into named columns:
      - ``annotation.<name>.score``
      - ``annotation.<name>.label``
      - ``annotation.<name>.text``
      - ``annotation.<name>.updated_at``
      - ``annotation.<name>.annotator_email``
      - ``annotation.<name>.annotator_id``

    Only non-None values are emitted. The original ``annotations`` key is
    removed from the row.
    """
    annotations = row.pop("annotations", None)
    if not annotations:
        return row
    for ann in annotations:
        name = ann.get("name")
        if not name:
            continue
        for field in ("score", "label", "text", "updated_at"):
            val = ann.get(field)
            if val is not None:
                row[f"{prefix}.{name}.{field}"] = val
        annotator = ann.get("annotator")
        if annotator:
            if annotator.get("email") is not None:
                row[f"{prefix}.{name}.annotator_email"] = annotator["email"]
            if annotator.get("id") is not None:
                row[f"{prefix}.{name}.annotator_id"] = annotator["id"]
    return row


def make_to_df(
    field_name: str, flatten_annotations: bool = False
) -> "Callable[..., pd.DataFrame]":
    """Return a ``to_df`` method bound to *field_name* on a list-response model.

    Args:
        field_name: Name of the list field on the response object (e.g.
            ``"users"``, ``"datasets"``).
        flatten_annotations: If ``True``, pivot the ``annotations`` list into
            per-annotation named columns.

    Returns:
        A ``to_df`` function suitable for assignment as a class method.
    """

    def to_df(
        self: object,
        by_alias: bool = False,
        exclude_none: str | bool = True,
        json_normalize: bool = False,
        convert_dtypes: bool = True,
        expand_field: str = "additional_properties",
        expand_prefix: str = "",
    ) -> "pd.DataFrame":
        """Convert a list of objects to a :class:`pandas.DataFrame`.

        Behavior:
          - If an item is a Pydantic v2 model, use `.model_dump(by_alias=...)`.
          - If an item is a mapping (dict-like), use it as-is.
          - Otherwise, raise a ValueError (unsupported row type).

        Args:
            self (object): The object instance containing the field to convert.
            by_alias (bool): Use field aliases when dumping Pydantic models.
            exclude_none (str | bool): Control None/NaN column dropping.
                - False: keep Nones as-is
                - "all": drop columns where all values are None/NaN
                - "any": drop columns where any value is None/NaN
                - True: alias for "all"
            json_normalize (bool): If True, flatten nested dicts via
                ``pandas.json_normalize``.
            convert_dtypes (bool): If True, call ``DataFrame.convert_dtypes()``
                at the end.
            expand_field (str): If set, look for this field in each row and
                expand its keys into top-level columns.
            expand_prefix (str): If set, prefix expanded column names with this
                string.

        Returns:
            pandas.DataFrame: The converted DataFrame.
        """
        import pandas as pd

        items = getattr(self, field_name, []) or []

        rows = []
        for it in items:
            if hasattr(it, "model_dump"):  # Pydantic v2 object
                row = it.model_dump(by_alias=by_alias)
            elif isinstance(it, Mapping):  # Plain mapping
                row = dict(it)  # Make a copy to avoid mutating the original
            else:
                raise ValueError(
                    f"Cannot convert item of type {type(it)} to DataFrame row"
                )

            # --- one-level expansion (no recursion) ---
            if expand_field and isinstance(row.get(expand_field), Mapping):
                nested = dict(row.pop(expand_field))
                if expand_prefix:
                    nested = {
                        f"{expand_prefix}{k}": v for k, v in nested.items()
                    }
                # nested keys win only if you want them to; swap order if not
                row = {**row, **nested}

            # --- flatten annotations list into named columns ---
            if flatten_annotations and "annotations" in row:
                row = _pivot_annotations(row)

            rows.append(row)

        df = (
            pd.json_normalize(rows, sep=".")
            if json_normalize
            else pd.DataFrame(rows)
        )

        # Drop None/NaN columns if requested
        if exclude_none in ("any", "all", True):
            drop_how: Literal["any", "all"] = (
                "all"
                if exclude_none is True
                else cast("Literal['any', 'all']", exclude_none)
            )
            df.dropna(axis=1, how=drop_how, inplace=True)

        if convert_dtypes:
            df = df.convert_dtypes()
        return df

    return to_df


def annotation_configs_to_df(
    self: object,
    by_alias: bool = False,
    exclude_none: str | bool = True,
    json_normalize: bool = False,
    convert_dtypes: bool = True,
    expand_field: str = "actual_instance",
    expand_prefix: str = "",
) -> "pd.DataFrame":
    """Convert annotation config list response to DataFrame.

    Defaults to expanding `actual_instance` so oneOf wrapper internals are
    flattened into user-facing columns.
    """
    to_df = make_to_df("annotation_configs")
    df = to_df(
        self,
        by_alias=by_alias,
        exclude_none=exclude_none,
        json_normalize=json_normalize,
        convert_dtypes=convert_dtypes,
        expand_field=expand_field,
        expand_prefix=expand_prefix,
    )
    return df.drop(
        columns=["one_of_schemas", "discriminator_value_class_map"],
        errors="ignore",
    )


# Monkey-patch convenience methods onto generated response models
# Type ignore comments needed: mypy can't verify runtime attribute additions
models.DatasetExampleListResponse.to_df = make_to_df(  # type: ignore[attr-defined]
    "examples", flatten_annotations=True
)
models.DatasetListResponse.to_df = make_to_df("datasets")  # type: ignore[attr-defined]
models.ExperimentListResponse.to_df = make_to_df("experiments")  # type: ignore[attr-defined]
models.ExperimentRunsListResponse.to_df = make_to_df(  # type: ignore[attr-defined]
    "experiment_runs", flatten_annotations=True
)
models.ProjectListResponse.to_df = make_to_df("projects")  # type: ignore[attr-defined]
models.AnnotationConfigListResponse.to_df = annotation_configs_to_df  # type: ignore[attr-defined]
models.SpanListResponse.to_df = make_to_df(  # type: ignore[attr-defined]
    "spans", flatten_annotations=True
)
models.SpaceListResponse.to_df = make_to_df("spaces")  # type: ignore[attr-defined]
models.ApiKeyListResponse.to_df = make_to_df("api_keys")  # type: ignore[attr-defined]
models.AnnotationQueueRecordListResponse.to_df = make_to_df("records")  # type: ignore[attr-defined]
models.AnnotationQueueListResponse.to_df = make_to_df("annotation_queues")  # type: ignore[attr-defined]
models.EvaluatorListResponse.to_df = make_to_df("evaluators")  # type: ignore[attr-defined]
models.EvaluatorVersionListResponse.to_df = make_to_df("evaluator_versions")  # type: ignore[attr-defined]
models.PromptListResponse.to_df = make_to_df("prompts")  # type: ignore[attr-defined]
models.PromptVersionListResponse.to_df = make_to_df("prompt_versions")  # type: ignore[attr-defined]
models.RoleListResponse.to_df = make_to_df("roles")  # type: ignore[attr-defined]
models.RoleBindingListResponse.to_df = make_to_df("role_bindings")  # type: ignore[attr-defined]
models.TaskListResponse.to_df = make_to_df("tasks")  # type: ignore[attr-defined]
models.TaskRunListResponse.to_df = make_to_df("task_runs")  # type: ignore[attr-defined]
models.AiIntegrationListResponse.to_df = make_to_df("ai_integrations")  # type: ignore[attr-defined]
models.OrganizationListResponse.to_df = make_to_df("organizations")  # type: ignore[attr-defined]
models.UserListResponse.to_df = make_to_df("users")  # type: ignore[attr-defined]

# Monkey-patch domain list-response types so .to_df() works on the
# SDK-typed objects returned by sub-clients (e.g. client.users.list()).
_DomainUserListResponse.to_df = make_to_df("users")  # type: ignore[attr-defined]
_DomainTaskListResponse.to_df = make_to_df("tasks")  # type: ignore[attr-defined]
_DomainEvaluatorVersionListResponse.to_df = make_to_df("evaluator_versions")  # type: ignore[attr-defined]
_DomainAnnotationConfigListResponse.to_df = make_to_df("annotation_configs")  # type: ignore[attr-defined]
