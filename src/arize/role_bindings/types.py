"""Public type re-exports for the role_bindings subdomain."""

from arize._generated.api_client.models.role_binding import RoleBinding
from arize._generated.api_client.models.role_binding_resource_type import (
    RoleBindingResourceType,
)

__all__ = [
    "RoleBinding",
    "RoleBindingResourceType",
]
