# src/arize/_lazy.py
from __future__ import annotations

import inspect
import logging
import sys
import threading
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import types

    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class LazySubclientsMixin:
    _SUBCLIENTS: ClassVar[dict[str, tuple[str, str]]] = {}
    _EXTRAS: ClassVar[dict[str, tuple[str | None, tuple[str, ...]]]] = {}

    def __init__(self, sdk_config: SDKConfiguration) -> None:
        self.sdk_config = sdk_config
        self._lazy_cache: dict[str, object] = {}
        self._lazy_lock = threading.Lock()

        # Add generated client factory
        from arize._client_factory import GeneratedClientFactory

        self._gen_client_factory = GeneratedClientFactory(sdk_config)

    def __getattr__(self, name: str) -> object:
        subs = self._SUBCLIENTS
        if name not in subs:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}"
            )

        with self._lazy_lock:
            if name in self._lazy_cache:
                return self._lazy_cache[name]

            logger.debug(f"Lazily loading subclient {name!r}")
            module_path, class_name = subs[name]
            extra_key, required = self._EXTRAS.get(name, (None, ()))
            require(extra_key, required)

            module = _dynamic_import(module_path)
            klass = getattr(module, class_name)

            # Determine which parameters this subclient needs
            # and build kwargs accordingly
            sig = inspect.signature(klass.__init__)
            kwargs: dict[str, object] = {}
            if "sdk_config" in sig.parameters:
                kwargs["sdk_config"] = self.sdk_config
            if "generated_client" in sig.parameters:
                kwargs["generated_client"] = (
                    self._gen_client_factory.get_client()
                )

            instance = klass(**kwargs)
            self._lazy_cache[name] = instance
            return instance

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *self._SUBCLIENTS.keys()})


class OptionalDependencyError(ImportError): ...


def _can_import(module_name: str) -> bool:
    """Check if a module can be imported without raising an exception.

    Args:
        module_name: The fully qualified module name to check (e.g., 'numpy', 'sklearn.preprocessing').

    Returns:
        bool: True if the module can be imported successfully, False otherwise.
    """
    try:
        import_module(module_name)
    except Exception:
        return False
    else:
        return True


def require(
    extra_key: str | None,
    required: tuple[str, ...],
    pkgname: str = "arize",
) -> None:
    """Ensure required optional dependencies are installed, raising an error if missing.

    Args:
        extra_key: The extras group key for pip install (e.g., 'mimic', 'embeddings').
            Used in the error message to guide users.
        required: Tuple of required module names to check for availability.
        pkgname: The package name for installation instructions. Defaults to 'arize'.

    Raises:
        OptionalDependencyError: If any of the required modules cannot be imported.
            The error message includes pip install instructions with the extras group.
    """
    if not required:
        return
    missing = [p for p in required if not _can_import(p)]
    if missing:
        raise OptionalDependencyError(
            f"Missing optional dependencies: {', '.join(missing)}. "
            f"Install via: pip install {pkgname}[{extra_key}]"
        )


def _dynamic_import(modname: str, retries: int = 2) -> types.ModuleType:
    """Dynamically import a module with retry logic and sys.modules cleanup on failure.

    Args:
        modname: The fully qualified module name to import.
        retries: Number of import attempts to make. Must be > 0. Defaults to 2.

    Returns:
        types.ModuleType: The successfully imported module.

    Raises:
        ValueError: If retries is <= 0.
        ModuleNotFoundError: If the module cannot be found after all retry attempts.
        ImportError: If the module import fails after all retry attempts.
        KeyError: If a key error occurs during import after all retry attempts.
    """

    def _attempt_import(remaining_attempts: int) -> types.ModuleType:
        try:
            return import_module(modname)
        except (ModuleNotFoundError, ImportError, KeyError):
            sys.modules.pop(modname, None)
            if remaining_attempts <= 1:
                raise
            return _attempt_import(remaining_attempts - 1)

    if retries <= 0:
        raise ValueError(f"retries must be > 0, got {retries}")
    return _attempt_import(retries)
