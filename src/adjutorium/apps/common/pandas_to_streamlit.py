# stdlib
from typing import Any, List

# third party
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype


class DropdownColumn:
    def __init__(self, name: str, val_range: list) -> None:
        self.type = "dropdown"
        self.name = name
        self.val_range = val_range


class CheckboxColumn:
    def __init__(self, name: str) -> None:
        self.type = "checkbox"
        self.name = name


class SliderInt:
    def __init__(self, name: str, minval: int, maxval: int) -> None:
        self.type = "slider_integer"
        self.name = name
        self.min = minval
        self.max = maxval
        self.mean = int((minval + maxval) / 2)


class SliderFloat:
    def __init__(self, name: str, minval: float, maxval: float) -> None:
        self.type = "slider_float"
        self.name = name
        self.min = float(minval)
        self.max = float(maxval)
        self.mean = float((minval + maxval) / 2)


def generate_menu(X: pd.DataFrame, checkboxes: List) -> list:
    dtype: Any
    column_types: List[Any] = []

    for col in X.columns:
        if is_integer_dtype(X[col].dtype):  # in ["int64", "integer", "int"]:
            unique_vals = X[col].unique()
            minval = X[col].min()
            maxval = X[col].max()
            if col in checkboxes:
                dtype = CheckboxColumn(col)
                column_types.append((col, dtype))
                continue

            if len(unique_vals) < 20:
                dtype = DropdownColumn(col, unique_vals)
                column_types.append((col, dtype))
                continue

            dtype = SliderInt(f"{col} ({int(minval)} - {int(maxval)})", minval, maxval)
            column_types.append((col, dtype))

        elif is_float_dtype(X[col].dtype):  # in ["float64", "float"]:
            unique_vals = X[col].unique()
            minval = X[col].min()
            maxval = X[col].max()

            if len(unique_vals) < 20:
                dtype = DropdownColumn(col, unique_vals)
                column_types.append((col, dtype))
                continue

            dtype = SliderFloat(
                f"{col} ({int(minval)} - {int(maxval)})", minval, maxval
            )
            column_types.append((col, dtype))
        else:
            unique_vals = X[col].unique()

            dtype = DropdownColumn(col, unique_vals)
            column_types.append((col, dtype))

    return column_types
