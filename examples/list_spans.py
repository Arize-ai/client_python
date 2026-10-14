"""List spans while excluding large embedding vectors."""

import os

from arize import ArizeClient


def main() -> None:
    """List spans without embedding vectors."""
    client = ArizeClient(api_key=os.environ["ARIZE_API_KEY"])
    response = client.spans.list(
        project=os.environ["ARIZE_PROJECT_ID"],
        excluded_columns=["attributes.embedding.vectors"],
    )
    if response.spans is None:
        raise ValueError("Server response omits spans")
    for span in response.spans:
        print(span.name)


if __name__ == "__main__":
    main()
