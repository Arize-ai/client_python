"""Unit tests for arize._flight.types module."""

from __future__ import annotations

import pytest

from arize._flight.types import FlightRequestType


@pytest.mark.unit
class TestFlightRequestType:
    """Unit tests for FlightRequestType enum."""

    @pytest.mark.parametrize(
        "member,expected_value",
        [
            (FlightRequestType.EVALUATION, "evaluation"),
            (FlightRequestType.ANNOTATION, "annotation"),
            (FlightRequestType.METADATA, "metadata"),
            (FlightRequestType.LOG_EXPERIMENT_DATA, "log_experiment_data"),
        ],
    )
    def test_enum_values(
        self, member: FlightRequestType, expected_value: str
    ) -> None:
        """Test that FlightRequestType member has expected value."""
        assert member == expected_value

    def test_enum_members_count(self) -> None:
        """Test that FlightRequestType has exactly 4 members."""
        assert len(FlightRequestType) == 4

    def test_enum_value_types(self) -> None:
        """Test that all enum values are strings."""
        for member in FlightRequestType:
            assert isinstance(member.value, str)

    def test_enum_comparison(self) -> None:
        """Test enum value comparisons."""
        assert FlightRequestType.EVALUATION == FlightRequestType.EVALUATION
        assert FlightRequestType.EVALUATION != FlightRequestType.ANNOTATION

    @pytest.mark.parametrize(
        "value,expected_member",
        [
            ("evaluation", FlightRequestType.EVALUATION),
            ("annotation", FlightRequestType.ANNOTATION),
            ("metadata", FlightRequestType.METADATA),
            ("log_experiment_data", FlightRequestType.LOG_EXPERIMENT_DATA),
        ],
    )
    def test_enum_by_value(
        self, value: str, expected_member: FlightRequestType
    ) -> None:
        """Test creating enum from string values."""
        assert FlightRequestType(value) == expected_member

    def test_enum_invalid_value(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            FlightRequestType("invalid_value")

    @pytest.mark.parametrize(
        "member",
        [
            FlightRequestType.EVALUATION,
            FlightRequestType.ANNOTATION,
            FlightRequestType.METADATA,
            FlightRequestType.LOG_EXPERIMENT_DATA,
        ],
    )
    def test_enum_members_list(self, member: FlightRequestType) -> None:
        """Test that expected member is present in enum."""
        members = list(FlightRequestType)
        assert member in members

    @pytest.mark.parametrize(
        "member,expected_name",
        [
            (FlightRequestType.EVALUATION, "EVALUATION"),
            (FlightRequestType.ANNOTATION, "ANNOTATION"),
            (FlightRequestType.METADATA, "METADATA"),
            (FlightRequestType.LOG_EXPERIMENT_DATA, "LOG_EXPERIMENT_DATA"),
        ],
    )
    def test_enum_name_attribute(
        self, member: FlightRequestType, expected_name: str
    ) -> None:
        """Test that enum member has correct name attribute."""
        assert member.name == expected_name

    @pytest.mark.parametrize(
        "member",
        [
            FlightRequestType.EVALUATION,
            FlightRequestType.ANNOTATION,
            FlightRequestType.METADATA,
            FlightRequestType.LOG_EXPERIMENT_DATA,
        ],
    )
    def test_enum_is_str_subclass(self, member: FlightRequestType) -> None:
        """Test that FlightRequestType enum value is string instance."""
        assert isinstance(member, str)
