from arize.utils.types import TypedValue


class CastingError(Exception):
    def __str__(self) -> str:
        return self.error_message()

    def __init__(self, error_msg: str, typed_value: TypedValue) -> None:
        self.error_msg = error_msg
        self.typed_value = typed_value

    def error_message(self) -> str:
        return (
            f"Failed to cast value {self.typed_value.value} of type {type(self.typed_value.value)} "
            f"to type {self.typed_value.type}. "
            f"Error: {self.error_msg}."
        )
