"""Download a dataset from Arize via Flight and save it to a parquet file.

Example::

    ARIZE_API_KEY_US=your-api-key \
        uv run --project sdk/python/arize/v8 \
        python sdk/python/arize/v8/tests/manual/test_download_dataset_via_flight.py \
        --dataset my-dataset --env prod --iteration 1

    ARIZE_API_KEY_DEV=your-api-key \
        uv run --project sdk/python/arize/v8 \
        python sdk/python/arize/v8/tests/manual/test_download_dataset_via_flight.py \
        --dataset my-dataset --file dataset_download.parquet \
        --env dev --iteration 1
"""

import argparse
import os
from collections.abc import Callable
from datetime import date
from pathlib import Path

from arize import ArizeClient

DEV_CLIENT_KWARGS = {
    "api_host": "devr.arize.com",
    "otlp_host": "devotlp.arize.com",
    "flight_host": "devx.arize.com",
    "flight_port": 443,
}
SPACE_IDS = {
    "prod": "YOUR-PROD-SPACE-ID-HERE",
    "dev": "YOUR-DEV-SPACE-ID-HERE",
}


def client_kwargs_for_environment(environment: str) -> dict[str, str | int]:
    """Return Arize client configuration for the selected environment."""
    api_key_name = (
        "ARIZE_API_KEY_DEV" if environment == "dev" else "ARIZE_API_KEY_US"
    )
    api_key = os.environ.get(api_key_name)
    if not api_key:
        raise ValueError(f"{api_key_name} environment variable is required")

    client_kwargs: dict[str, str | int] = {
        "api_key": api_key,
        "pyarrow_max_chunksize": 3_000,
    }
    if environment == "dev":
        client_kwargs.update(DEV_CLIENT_KWARGS)
    return client_kwargs


def resolve_output_file(
    dataset_name: str,
    output_file: Path | None,
    prompt: Callable[[str], str] | None = None,
) -> Path | None:
    """Return the output file when it is safe to write."""
    output_file = output_file or Path(f"{dataset_name}.parquet")
    if not output_file.exists():
        return output_file

    confirm = prompt or input
    response = confirm(f"{output_file} already exists. Overwrite? [y/N] ")
    if response.strip().lower() in {"y", "yes"}:
        return output_file
    return None


def main(argv: list[str] | None = None, *, run_date: date | None = None) -> int:
    """Download an Arize dataset to a parquet file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--env", choices=("prod", "dev"), required=True)
    args = parser.parse_args(argv)

    client = ArizeClient(**client_kwargs_for_environment(args.env))
    sdk_config = client.sdk_config
    print(
        f"DEBUGPRINT: test_download_dataset_via_flight.py:74: sdk_config={sdk_config}"
    )

    space_id = SPACE_IDS[args.env]
    space = client.spaces.get(space=space_id)
    print(f"DEBUGPRINT: test_download_dataset_via_flight.py:80: space={space}")

    dataset = client.datasets.get(space=space, dataset=args.dataset)
    print(
        f"DEBUGPRINT: test_download_dataset_via_flight.py:82: dataset={dataset}"
    )

    output_file = resolve_output_file(dataset.name, args.file)
    if output_file is None:
        print("Download canceled.")
        return 0

    full_dataset = client.datasets.list_examples(dataset=dataset.id, all=True)
    full_dataset_df = full_dataset.to_df()

    N = len(full_dataset_df)
    print(f"DEBUGPRINT: test_download_dataset_via_flight.py:90: N={N}")
    cols = full_dataset_df.columns
    print(f"DEBUGPRINT: test_download_dataset_via_flight.py:92: cols={cols}")

    print(full_dataset_df.head())

    full_dataset_df.to_parquet(output_file, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
