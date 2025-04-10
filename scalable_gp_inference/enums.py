from enum import Enum, auto


class _StackingMode(Enum):
    """Enumeration for different stacking modes.

    Attributes:
        HORIZONTAL: Horizontal stacking mode.
        VERTICAL: Vertical stacking mode.
    """

    HORIZONTAL = auto()
    VERTICAL = auto()

    @classmethod
    def _from_str(cls, value, param_name):
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value = value.lower()
            if value == "horizontal":
                return cls.HORIZONTAL
            elif value == "vertical":
                return cls.VERTICAL

        raise ValueError(
            f"Invalid value for {param_name}: {value}. "
            "Expected 'horizontal', 'vertical', _StackingMode.HORIZONTAL, "
            "or _StackingMode.VERTICAL."
        )
