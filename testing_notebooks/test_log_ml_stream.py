import os

from arize import ArizeClient
from arize.types import (
    ArizeTypes,
    Embedding,
    Environments,
    ModelTypes,
    TypedValue,
)

SPACE_ID = "U3BhY2U6NTA3MDpsTlIr"
API_KEY = (
    "ak-ed52c99c-19ac-4f89-b43b-8c650d6771c9-KdAJlHVKJxS-ZTbc8xlAPsEo-7ZFCPIC"
)
MODEL_NAME = "test-sdkv8-stream-09-30-25-c"


def main():
    os.environ["ARIZE_LOG_ENABLE"] = "true"
    os.environ["ARIZE_LOG_LEVEL"] = "debug"
    os.environ["ARIZE_LOG_STRUCTURED"] = "false"

    client = ArizeClient(api_key=API_KEY)
    print("arize client", client)

    # Example features; features & tags can be optionally defined with typing
    features = {
        "state": "ca",
        "city": "berkeley",
        "merchant_name": "Peets Coffee",
        "pos_approved": TypedValue(value=False, type=ArizeTypes.INT),
        "item_count": 10,
        "merchant_type": "coffee shop",
        "charge_amount": TypedValue(value=20.11, type=ArizeTypes.FLOAT),
    }

    # Example embedding features
    embedding_features = {
        "nlp_embedding": Embedding(
            vector=[4.0, 5.0, 6.0, 7.0],
            data="This is a test sentence",
        ),
    }

    response = client.models.log_stream(
        space_id=SPACE_ID,
        model_name=MODEL_NAME,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        prediction_label=("not fraud", 0.3),
        actual_label=("fraud", 1.0),
        features=features,
        embedding_features=embedding_features,
    )

    print("response", response)
    print("response.result()", response.result())


if __name__ == "__main__":
    main()
