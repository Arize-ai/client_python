"""Arize SDK for model observability and LLM tracing."""

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, cast

from arize._generated.api_client import models
from arize.client import ArizeClient
from arize.config import SDKConfiguration
from arize.regions import Region

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
    "ArizeClient",
    "Region",
    "SDKConfiguration",
]


def make_to_df(field_name: str) -> Callable[..., "pd.DataFrame"]:
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
            json_normalize (bool): If True, flatten nested dicts via `pandas.json_normalize`.
            convert_dtypes (bool): If True, call `DataFrame.convert_dtypes()` at the end.
            expand_field (str): If set, look for this field in each row and
            expand its keys into top-level columns.
            expand_prefix (str): If set, prefix expanded column names with this string.

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
models.DatasetsExamplesList200Response.to_df = make_to_df("examples")  # type: ignore[attr-defined]
models.DatasetsList200Response.to_df = make_to_df("datasets")  # type: ignore[attr-defined]
models.ExperimentsList200Response.to_df = make_to_df("experiments")  # type: ignore[attr-defined]
models.ExperimentsRunsList200Response.to_df = make_to_df("experiment_runs")  # type: ignore[attr-defined]
models.ProjectsList200Response.to_df = make_to_df("projects")  # type: ignore[attr-defined]
models.AnnotationConfigsList200Response.to_df = annotation_configs_to_df  # type: ignore[attr-defined]
models.SpansList200Response.to_df = make_to_df("spans")  # type: ignore[attr-defined]
