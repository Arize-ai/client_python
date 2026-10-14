"""Regression checks for the OpenAPI response post-generation script."""

import runpy
from pathlib import Path

import pytest


@pytest.fixture
def patcher() -> dict:
    """Load the script without reading a spec or writing generated files."""
    script = next(
        (
            parent / "scripts/openapi/patch_forward_compat_models.py"
            for parent in Path(__file__).resolve().parents
            if (
                parent / "scripts/openapi/patch_forward_compat_models.py"
            ).is_file()
        ),
        None,
    )
    if script is None:
        pytest.skip("OpenAPI generation checks require the Arize monorepo")
    return runpy.run_path(str(script))


@pytest.fixture
def generated_source() -> str:
    """Return a model emitted by the pinned OpenAPI generator."""
    return (
        Path(__file__).parent / "fixtures/generated_agent_config.py.txt"
    ).read_text()


@pytest.mark.unit
def test_generated_model_rewrite(patcher, generated_source) -> None:
    """Rewriting preserves constraints, extra fields, and nested responses."""
    text = patcher["make_response_fields_optional"](generated_source)
    namespace = {}
    exec(compile(text, "generated_agent_config.py", "exec"), namespace)  # noqa: S102
    model = namespace["AgentConfig"]
    # from __future__ annotations need a namespace for resolving fixture imports.
    model.model_rebuild(_types_namespace=namespace)
    response = model.from_dict(
        {"future_field": "kept", "request_presets": [{"name": "preset"}]}
    )
    assert response.endpoint is None
    assert response.request_presets[0].config is None
    assert response.additional_properties == {"future_field": "kept"}
    with pytest.raises(ValueError):
        model.from_dict({"endpoint": "x" * 2049})


@pytest.mark.unit
def test_changed_generator_template_fails(patcher, generated_source) -> None:
    """Unexpected validation structure is reported before writing bad models."""
    source = generated_source.replace("cls.model_validate", "cls.parse_obj")
    with pytest.raises(RuntimeError, match="validation dictionary"):
        patcher["make_response_fields_optional"](source)


@pytest.mark.unit
def test_response_guard_catches_additional_properties(
    patcher, generated_source, tmp_path
) -> None:
    """An unmarked extensible response cannot bypass the coverage guard."""
    (tmp_path / "agent_config.py").write_text(generated_source)
    spec = {
        "components": {"schemas": {"AgentConfig": {"type": "object"}}},
        "paths": {
            "/agents": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AgentConfig"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    }
    with pytest.raises(RuntimeError, match="AgentConfig"):
        patcher["patch_models"](spec, tmp_path)


@pytest.mark.unit
def test_request_marker_is_rejected(patcher, tmp_path) -> None:
    """A schema used in requests must retain strict validation."""
    spec = {
        "components": {"schemas": {"Shared": {"x-forward-compatible": True}}},
        "paths": {
            "/shared": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/Shared"
                                }
                            }
                        }
                    }
                }
            }
        },
    }
    with pytest.raises(RuntimeError, match="from requests: Shared"):
        patcher["patch_models"](spec, tmp_path)


@pytest.mark.unit
def test_response_union_requires_mapping(patcher) -> None:
    """Future optional variants cannot use ambiguous fallback matching."""
    with pytest.raises(RuntimeError, match="requires a discriminator mapping"):
        patcher["patch_union"]("", {"oneOf": [{"$ref": "#/Variant"}]}, "Union")


@pytest.mark.unit
def test_response_union_rejects_unknown_mapping_target(patcher) -> None:
    """A discriminator cannot dispatch to a class outside the union."""
    schema = {
        "oneOf": [{"$ref": "#/Variant"}],
        "discriminator": {
            "propertyName": "type",
            "mapping": {"A": "#/Unknown"},
        },
    }
    with pytest.raises(RuntimeError, match="unsupported discriminator mapping"):
        patcher["patch_union"]("", schema, "Union")
