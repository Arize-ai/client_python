class Validator:
    @staticmethod
    def validate_input_type(input, input_name: str, input_type: type) -> None:
        if not isinstance(input, input_type) and input is not None:
            raise TypeError(
                f"{input_name} {input} is type {type(input)}, but must be a {input_type.__name__}"
            )

    @staticmethod
    def validate_input_value(input, input_name: str, choices: tuple) -> None:
        if input not in choices:
            raise ValueError(f"{input_name} is {input}, but must be one of {', '.join(choices)}")
