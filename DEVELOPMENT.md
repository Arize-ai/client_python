# Developing the Arize Python SDK (v8)

This document is for contributors working in the `sdk/python/arize/v8/` package. 

For repository layout and conventions, see [AGENTS.md](AGENTS.md).

## Setup

- **Python:** 3.10+ (see `pyproject.toml`).
- **Package manager:** [uv](https://docs.astral.sh/uv/). From this directory:

  ```bash
  uv sync
  ```

- Run tooling via **`task`** ([taskipy](https://github.com/iamgodoy/taskipy)), e.g. `uv run task lint`, or invoke commands from `pyproject.toml` directly.

## Daily workflow

Run from `sdk/python/arize/v8/`:

| Command | Purpose |
|---------|---------|
| `task lint` | `ruff format` + `ruff check --fix` |
| `task type-check` | `mypy` |
| `task test` | Unit tests (`tests/unit/`) with coverage (90% threshold on non-excluded code) |
| `task test-integration` | Integration tests (`tests/integration/`)  |

After substantial changes, run **lint**, **type-check**, and **test** before opening a PR.

## Integration tests

Integration tests live under `tests/integration/`, call the **real** Arize API, and are marked with `@pytest.mark.integration`. They are **skipped** unless credentials are set.

### Required for most integration tests

| Variable | Purpose |
|----------|---------|
| `ARIZE_API_KEY` | API key with access to the test space |
| `ARIZE_TEST_SPACE_NAME` | Space **name** or base64 space **ID** to run against |

```bash
export ARIZE_API_KEY="..."
export ARIZE_TEST_SPACE_NAME="your-space-name"
uv run task test-integration
```

### Per-module notes

- **Evaluators** (`test_evaluators_flows.py`): Templates use an **AI integration** in the test space (the first integration from `ai_integrations.list`). That integration must have **working provider credentials** (for the default model in the tests, an OpenAI integration needs a real OpenAI API key configured **on the integration in Arize**). The API also requires the template body to include at least one f-string-style placeholder (e.g. `{output}`), not `{{...}}`. If the space has no integrations, those tests skip. See the module docstring in that file for details.

- **Projects** (`test_projects_flows.py`): Cleanup calls `projects.delete` when permitted. If the key cannot delete projects, delete is ignored so assertions can still pass; test projects may remain in the space.

- **Roles** (`test_roles_flows.py`): Creating or mutating roles requires an **account admin** API key. Set `ARIZE_TEST_ACCOUNT_ADMIN_FLOWS=1` and use an admin key; otherwise the module is skipped.

## Generated code

Do not edit `src/arize/_generated/` by hand. Regenerate from the repo root:

| Artifact | Script |
|----------|--------|
| OpenAPI REST client | `./scripts/recompile_openapi.sh` |
| Protobuf | `./scripts/recompile_protos.sh` |

Ruff and mypy exclude generated paths (see `pyproject.toml`).
