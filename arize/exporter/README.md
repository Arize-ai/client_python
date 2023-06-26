## Arize Python Exporter Client - User Guide

### Step 1: Pip install `arize` and set up `api_key` and `space_id`<br>
```
! pip install -q arize
```
```
api_key = '<arize_api_key>'
space_id = '<space_id>'
```
- You can get your `space_id` by visiting [app.arize.com](https://app.arize.com). The url will be in this format: `https://app.arize.com/organizations/:org_id/spaces/:space_id` <br>
  **NOTE: this is not the same as the space key used to send data using the SDK** <br>

- To get `api_key`, you must have Developer Access to your space. Visit [arize docs](https://docs.arize.com/arize/integrations/graphql-api/getting-started-with-programmatic-access) for more details <br>
  **NOTE: this is not the same as the api key in Space Settings** <br>

### Step 2: Initiate an `ArizeExportClient`<br>

```
from arize.exporter import ArizeExportClient

client = ArizeExportClient(api_key=api_key)
```

### Step 3: Export production data with predictions only to a pandas dataframe

```
from arize.utils.types import Environments
from datetime import datetime

start_time = datetime(2023, 4, 10, 0, 0, 0, 0)
end_time = datetime(2023, 4, 15, 0, 0, 0, 0)

df = client.export_model_to_df(
    space_id=space_id,
    model_id='arize-demo-fraud-use-case',
    environment=Environments.PRODUCTION,
    start_time=start_time,
    end_time=end_time,
    model_version='<model_version>',  #optional field
    batch_id='<batch_id>',  #optional field
)
```

### Export production data with predictions and actuals to a pandas dataframe

```
from arize.utils.types import Environments
from datetime import datetime

start_time = datetime(2023, 4, 10, 0, 0, 0, 0)
end_time = datetime(2023, 4, 15, 0, 0, 0, 0)

df = client.export_model_to_df(
    space_id=<space_id>,
    model_id='arize-demo-fraud-use-case',
    environment=Environments.PRODUCTION,
    start_time=start_time,
    end_time=end_time,
    include_actuals=True,  #optional field
    model_version='<model_version>',  #optional field
    batch_id='<batch_id>',  #optional field
)
```


### Export training data to a pandas dataframe

```
from arize.utils.types import Environments
from datetime import datetime

start_time = datetime(2023, 4, 10, 0, 0, 0, 0)
end_time = datetime(2023, 4, 15, 0, 0, 0, 0)

df = client.export_model_to_df(
    space_id=<space_id>,
    model_id='arize-demo-fraud-use-case',
    environment=Environments.TRAINING,
    start_time=start_time,
    end_time=end_time,
    model_version='<model_version>',  #optional field
    batch_id='<batch_id>',  #optional field
 )
```
### Export validation data to a pandas dataframe

```
from arize.utils.types import Environments
from datetime import datetime

start_time = datetime(2023, 4, 10, 0, 0, 0, 0)
end_time = datetime(2023, 4, 15, 0, 0, 0, 0)

df = client.export_model_to_df(
    space_id=<space_id>,
    model_id='arize-demo-fraud-use-case',
    environment=Environments.VALIDATION,
    start_time=start_time,
    end_time=end_time,
    model_version='<model_version>',  #optional field
    batch_id='<batch_id>',  #optional field
)
```

### Export production data to a parquet file

```
from arize.utils.types import Environments
from datetime import datetime

start_time = datetime(2023, 4, 10, 0, 0, 0, 0)
end_time = datetime(2023, 4, 15, 0, 0, 0, 0)

client.export_model_to_parquet(
    path = "example.parquet",
    space_id=<space_id>,
    model_id='arize-demo-fraud-use-case',
    environment=Environments.PRODUCTION,
    start_time=start_time,
    end_time=end_time,
    model_version='<model_version>',  #optional field
    batch_id='<batch_id>',  #optional field
)

df = pd.read_parquet("example.parquet")
```
