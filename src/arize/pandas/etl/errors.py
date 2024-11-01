from arize.utils.logging import log_a_list
from arize.utils.types import TypedColumns


class ColumnCastingError(Exception):
    def __str__(self) -> str:
        return self.error_message()

    def __init__(
        self,
        error_msg: str,
        attempted_columns: str,
        attempted_type: TypedColumns,
    ) -> None:
        self.error_msg = error_msg
        self.attempted_casting_columns = attempted_columns
        self.attempted_casting_type = attempted_type

    def error_message(self) -> str:
        return (
            f"Failed to cast to type {self.attempted_casting_type} "
            f"for columns: {log_a_list(self.attempted_casting_columns, 'and')}. "
            f"Error: {self.error_msg}"
        )


class InvalidTypedColumnsError(Exception):
    def __str__(self) -> str:
        return self.error_message()

    def __init__(self, field_name: str, reason: str) -> None:
        self.field_name = field_name
        self.reason = reason

    def error_message(self) -> str:
        return f"The {self.field_name} TypedColumns object {self.reason}."


class InvalidSchemaFieldTypeError(Exception):
    def __str__(self) -> str:
        return self.error_message()

    def __init__(self, msg: str) -> None:
        self.msg = msg

    def error_message(self) -> str:
        return self.msg
