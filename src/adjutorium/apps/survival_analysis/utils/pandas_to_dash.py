# stdlib
from typing import Any, List, Tuple

# third party
from dash import html
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

# adjutorium absolute
import adjutorium.apps.survival_analysis.utils.dash_reusable_components as drc


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


def generate_menu(X: pd.DataFrame, checkboxes: List) -> Tuple:
    print("checkboxes", checkboxes)
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

    children = [
        html.H4(
            "Patient information", style={"font-size": "20px", "font-family": "inherit"}
        )
    ]
    for col, obj in column_types:
        if obj.type == "checkbox":
            col_layout = drc.NamedCheckbox(
                name=obj.name,
            )
            children.append(col_layout)
        if obj.type == "dropdown":
            col_layout = drc.NamedDropdown(
                name=obj.name,
                options=[{"label": val, "value": val} for val in obj.val_range],
            )
            children.append(col_layout)
        elif obj.type == "slider_integer":
            col_layout = drc.NamedInput(
                obj.name,
                min=obj.min,
                max=obj.max,
            )
            children.append(col_layout)
        elif obj.type == "slider_float":
            col_layout = drc.NamedInput(
                obj.name,
                min=obj.min,
                max=obj.max,
                step=0.1,
            )
            children.append(col_layout)

    return children, column_types
