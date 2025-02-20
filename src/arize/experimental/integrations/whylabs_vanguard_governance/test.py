import os

from arize.experimental.integrations.whylabs_vanguard_governance import (
    IntegrationClient,
)
from arize.utils.types import Environments, ModelTypes

os.environ["ARIZE_SPACE_ID"] = "REPLACE_ME"
os.environ["ARIZE_API_KEY"] = "REPLACE_ME"
os.environ["ARIZE_DEVELOPER_KEY"] = "REPLACE_ME"


def test_create_model():
    client = IntegrationClient(
        api_key=os.environ["ARIZE_API_KEY"],
        space_id=os.environ["ARIZE_SPACE_ID"],
        developer_key=os.environ["ARIZE_DEVELOPER_KEY"],
    )
    client.create_model(
        model_id="test-create-binary-classification",
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.BINARY_CLASSIFICATION,
    )


if __name__ == "__main__":
    test_create_model()
