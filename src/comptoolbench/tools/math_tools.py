"""Math and data analysis tools: percentage, rounding, statistics, regression.

These tools are fully local — no API keys needed. All functions are
pure and deterministic, so live and simulated modes are identical.
"""

from __future__ import annotations

import math
from typing import Any

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)


@register_tool
class PercentageChange(BaseTool):
    """Calculate the percentage change between two numbers."""

    name = "percentage_change"
    schema = ToolSchema(
        name="percentage_change",
        description="Calculate the percentage change between an old value and a new value. Returns the percent change and whether it was an increase, decrease, or unchanged.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="old_value",
                type="number",
                description="The original value (baseline)",
            ),
            ToolParameter(
                name="new_value",
                type="number",
                description="The new value to compare against the original",
            ),
        ],
        returns="The percentage change with direction (increase/decrease/unchanged)",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Compute percentage change from old_value to new_value."""
        old_value = float(kwargs["old_value"])
        new_value = float(kwargs["new_value"])

        if old_value == 0:
            raise ValueError("Cannot compute percentage change from zero")

        change_percent = ((new_value - old_value) / abs(old_value)) * 100

        if change_percent > 0:
            direction = "increase"
        elif change_percent < 0:
            direction = "decrease"
        else:
            direction = "unchanged"

        return {
            "change_percent": round(change_percent, 4),
            "direction": direction,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Percentage change is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class RoundNumber(BaseTool):
    """Round a number to a specified precision."""

    name = "round_number"
    schema = ToolSchema(
        name="round_number",
        description="Round a number to a specified number of decimal places. Defaults to 2 decimal places if not specified.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="value",
                type="number",
                description="The number to round",
            ),
            ToolParameter(
                name="decimals",
                type="integer",
                description="Number of decimal places (default 2)",
                required=False,
                default=2,
            ),
        ],
        returns="The rounded number",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Round value to the given number of decimal places."""
        value = float(kwargs["value"])
        decimals = int(kwargs.get("decimals", 2))

        if decimals < 0:
            raise ValueError("Decimal places must be non-negative")

        return {"rounded": round(value, decimals)}

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Rounding is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class MinMax(BaseTool):
    """Find the minimum and maximum values in a list."""

    name = "min_max"
    schema = ToolSchema(
        name="min_max",
        description="Find the minimum and maximum values in a list of numbers and compute the range (max - min).",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="numbers",
                type="array",
                description="List of numbers to find min and max of",
                items={"type": "number"},
            ),
        ],
        returns="The minimum, maximum, and range of the list",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Find min, max, and range of the number list."""
        numbers = [float(n) for n in kwargs["numbers"]]

        if not numbers:
            raise ValueError("Cannot find min/max of empty list")

        min_val = min(numbers)
        max_val = max(numbers)

        return {
            "min": min_val,
            "max": max_val,
            "range": round(max_val - min_val, 6),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Min/max is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class MovingAverage(BaseTool):
    """Compute the moving average of a list of values."""

    name = "moving_average"
    schema = ToolSchema(
        name="moving_average",
        description="Compute the simple moving average of a list of numbers using a specified window size. Returns a list of averages, one for each complete window.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="values",
                type="array",
                description="List of numbers to compute moving average over",
                items={"type": "number"},
            ),
            ToolParameter(
                name="window",
                type="integer",
                description="The window size for the moving average",
            ),
        ],
        returns="List of moving averages for each complete window",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Compute simple moving average with the given window."""
        values = [float(v) for v in kwargs["values"]]
        window = int(kwargs["window"])

        if window <= 0:
            raise ValueError("Window size must be positive")
        if window > len(values):
            raise ValueError(
                f"Window size ({window}) exceeds number of values ({len(values)})"
            )

        averages: list[float] = []
        for i in range(len(values) - window + 1):
            window_slice = values[i : i + window]
            avg = sum(window_slice) / window
            averages.append(round(avg, 4))

        return {"averages": averages}

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Moving average is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class NormalizeData(BaseTool):
    """Normalize a list of numbers to the 0-1 range (min-max scaling)."""

    name = "normalize_data"
    schema = ToolSchema(
        name="normalize_data",
        description="Normalize a list of numbers to the 0-1 range using min-max scaling. Returns the normalized values along with the original min and max used for scaling.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="values",
                type="array",
                description="List of numbers to normalize",
                items={"type": "number"},
            ),
        ],
        returns="Normalized values (0-1 range) with original min and max",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Normalize values using min-max scaling to [0, 1]."""
        values = [float(v) for v in kwargs["values"]]

        if not values:
            raise ValueError("Cannot normalize an empty list")

        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val

        if val_range == 0:
            # All values identical — normalize to 0.0
            normalized = [0.0] * len(values)
        else:
            normalized = [
                round((v - min_val) / val_range, 6) for v in values
            ]

        return {
            "normalized": normalized,
            "min_val": min_val,
            "max_val": max_val,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Normalization is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class Correlation(BaseTool):
    """Compute the Pearson correlation coefficient between two lists."""

    name = "correlation"
    schema = ToolSchema(
        name="correlation",
        description="Compute the Pearson correlation coefficient between two lists of numbers. Returns a value between -1 and 1 along with a human-readable interpretation.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="x",
                type="array",
                description="First list of numbers",
                items={"type": "number"},
            ),
            ToolParameter(
                name="y",
                type="array",
                description="Second list of numbers (must be same length as x)",
                items={"type": "number"},
            ),
        ],
        returns="Pearson correlation coefficient with interpretation",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Compute Pearson r between x and y."""
        x = [float(v) for v in kwargs["x"]]
        y = [float(v) for v in kwargs["y"]]

        if len(x) != len(y):
            raise ValueError(
                f"Lists must have equal length (got {len(x)} and {len(y)})"
            )
        if len(x) < 2:
            raise ValueError("Need at least 2 data points for correlation")

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        if std_x == 0 or std_y == 0:
            raise ValueError(
                "Cannot compute correlation: one or both variables have zero variance"
            )

        r = cov / (std_x * std_y)
        r = round(r, 6)

        abs_r = abs(r)
        if abs_r >= 0.9:
            strength = "very strong"
        elif abs_r >= 0.7:
            strength = "strong"
        elif abs_r >= 0.5:
            strength = "moderate"
        elif abs_r >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"

        direction = "positive" if r > 0 else "negative" if r < 0 else "no"
        interpretation = f"{strength} {direction} correlation"

        return {
            "correlation": r,
            "interpretation": interpretation,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Correlation is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class LinearRegression(BaseTool):
    """Fit a simple linear regression (y = mx + b)."""

    name = "linear_regression"
    schema = ToolSchema(
        name="linear_regression",
        description="Fit a simple linear regression model (y = mx + b) to two lists of numbers. Returns the slope, intercept, and R-squared goodness-of-fit measure.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="x",
                type="array",
                description="Independent variable values",
                items={"type": "number"},
            ),
            ToolParameter(
                name="y",
                type="array",
                description="Dependent variable values (must be same length as x)",
                items={"type": "number"},
            ),
        ],
        returns="Slope, intercept, and R-squared of the fitted line",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Fit y = mx + b via ordinary least squares."""
        x = [float(v) for v in kwargs["x"]]
        y = [float(v) for v in kwargs["y"]]

        if len(x) != len(y):
            raise ValueError(
                f"Lists must have equal length (got {len(x)} and {len(y)})"
            )
        if len(x) < 2:
            raise ValueError("Need at least 2 data points for regression")

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        ss_xx = sum((xi - mean_x) ** 2 for xi in x)
        ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        ss_yy = sum((yi - mean_y) ** 2 for yi in y)

        if ss_xx == 0:
            raise ValueError(
                "Cannot fit regression: all x values are identical"
            )

        slope = ss_xy / ss_xx
        intercept = mean_y - slope * mean_x

        # R-squared: proportion of variance explained
        if ss_yy == 0:
            r_squared = 1.0  # Perfect fit (constant y predicted by constant x)
        else:
            r_squared = (ss_xy**2) / (ss_xx * ss_yy)

        return {
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "r_squared": round(r_squared, 6),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Linear regression is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class Percentile(BaseTool):
    """Compute the percentile rank of a value within a list."""

    name = "percentile"
    schema = ToolSchema(
        name="percentile",
        description="Compute the percentile rank of a given value within a list of numbers. The percentile indicates the percentage of values in the list that are less than or equal to the given value.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="values",
                type="array",
                description="List of numbers defining the distribution",
                items={"type": "number"},
            ),
            ToolParameter(
                name="value",
                type="number",
                description="The value whose percentile rank to compute",
            ),
        ],
        returns="The percentile rank (0-100) of the value",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Compute percentile rank of value in the distribution."""
        values = [float(v) for v in kwargs["values"]]
        value = float(kwargs["value"])

        if not values:
            raise ValueError("Cannot compute percentile of empty list")

        count_below = sum(1 for v in values if v <= value)
        pct = (count_below / len(values)) * 100

        return {"percentile": round(pct, 4)}

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Percentile is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class StandardDeviation(BaseTool):
    """Compute the standard deviation of a list of numbers."""

    name = "standard_deviation"
    schema = ToolSchema(
        name="standard_deviation",
        description="Compute the mean, sample standard deviation, and variance of a list of numbers.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="values",
                type="array",
                description="List of numbers to compute standard deviation of",
                items={"type": "number"},
            ),
        ],
        returns="Mean, standard deviation, and variance of the values",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Compute mean, sample std dev, and variance."""
        values = [float(v) for v in kwargs["values"]]

        if not values:
            raise ValueError("Cannot compute standard deviation of empty list")
        if len(values) < 2:
            raise ValueError(
                "Need at least 2 values for sample standard deviation"
            )

        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std_dev = math.sqrt(variance)

        return {
            "mean": round(mean, 6),
            "std_dev": round(std_dev, 6),
            "variance": round(variance, 6),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Standard deviation is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class ClampValue(BaseTool):
    """Clamp a number within a specified range."""

    name = "clamp_value"
    schema = ToolSchema(
        name="clamp_value",
        description="Clamp a number so it falls within the specified minimum and maximum range. Returns the clamped value and whether clamping was applied.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="value",
                type="number",
                description="The number to clamp",
            ),
            ToolParameter(
                name="min_val",
                type="number",
                description="The minimum allowed value",
            ),
            ToolParameter(
                name="max_val",
                type="number",
                description="The maximum allowed value",
            ),
        ],
        returns="The clamped value and whether clamping was applied",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Clamp value to [min_val, max_val]."""
        value = float(kwargs["value"])
        min_val = float(kwargs["min_val"])
        max_val = float(kwargs["max_val"])

        if min_val > max_val:
            raise ValueError(
                f"min_val ({min_val}) must be <= max_val ({max_val})"
            )

        clamped = max(min_val, min(value, max_val))
        was_clamped = clamped != value

        return {
            "clamped": clamped,
            "was_clamped": was_clamped,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Clamping is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


__all__ = [
    "ClampValue",
    "Correlation",
    "LinearRegression",
    "MinMax",
    "MovingAverage",
    "NormalizeData",
    "Percentile",
    "PercentageChange",
    "RoundNumber",
    "StandardDeviation",
]
