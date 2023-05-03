class InvalidIndexError(Exception):
    def __repr__(self) -> str:
        return "Invalid_Index_Error"

    def __str__(self) -> str:
        return self.error_message()

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name

    def error_message(self) -> str:
        if self.field_name == "DataFrame":
            return (
                f"The index of the {self.field_name} is invalid; "
                f"reset the index by using df.reset_index(drop=True, inplace=True)"
            )
        else:
            return (
                f"The index of the Series given by the column '{self.field_name}' is invalid; "
                f"reset the index by using df.reset_index(drop=True, inplace=True)"
            )
