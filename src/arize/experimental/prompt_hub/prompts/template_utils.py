"""
Utility functions for delimiter-based template formatting.
"""

from typing import Any, Mapping

from arize.experimental.prompt_hub.prompts.base import PromptInputVariableFormat


def format_with_delimiters(
    template: str,
    variables: Mapping[str, Any],
    start_delim: str = "{{",
    end_delim: str = "}}",
) -> str:
    """
    Format a template string using delimiter-based variable substitution.

    Args:
        template: The template string containing variables
        variables: Dictionary of variable names to values
        start_delim: Start delimiter for variables (default: "{{")
        end_delim: End delimiter for variables (default: "}}")

    Returns:
        Formatted string with variables replaced

    Example:
        >>> template = "Hello {{name}}, you are {{age}} years old"
        >>> variables = {"name": "Alice", "age": 30}
        >>> format_with_delimiters(template, variables)
        'Hello Alice, you are 30 years old'
    """
    formatted = template
    for variable_name, value in variables.items():
        placeholder = start_delim + variable_name + end_delim
        formatted = formatted.replace(placeholder, str(value))
    return formatted


def format_with_f_string_style(
    template: str, variables: Mapping[str, Any]
) -> str:
    """
    Format a template string using f-string style variable substitution.

    Args:
        template: The template string containing variables
        variables: Dictionary of variable names to values

    Returns:
        Formatted string with variables replaced

    Example:
        >>> template = "Hello {name}, you are {age} years old"
        >>> variables = {"name": "Alice", "age": 30}
        >>> format_with_f_string_style(template, variables)
        'Hello Alice, you are 30 years old'
    """
    try:
        return template.format(**variables)
    except KeyError as e:
        # If a variable is missing, replace it with the placeholder
        missing_var = str(e).strip("'")
        return template.replace(f"{{{missing_var}}}", f"{{{missing_var}}}")


def format_template(
    template: str,
    variables: Mapping[str, Any],
    format_type: PromptInputVariableFormat,
) -> str:
    """
    Format a template string using the specified format type.

    Args:
        template: The template string containing variables
        variables: Dictionary of variable names to values
        format_type: PromptInputVariableFormat enum value

    Returns:
        Formatted string with variables replaced
    """
    if format_type == PromptInputVariableFormat.MUSTACHE:
        return format_with_delimiters(template, variables, "{{", "}}")
    elif format_type == PromptInputVariableFormat.F_STRING:
        return format_with_f_string_style(template, variables)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
