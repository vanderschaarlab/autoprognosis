# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder


class DatetimeEncoder:
    """Datetime encoder, with sklearn-style API"""

    def __init__(self) -> None:
        pass

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.Series) -> Any:
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, X: pd.Series) -> pd.Series:
        out = pd.to_numeric(X).astype(float)
        return out

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, X: pd.Series) -> pd.Series:
        out = pd.to_datetime(X)
        return out

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform(self, X: pd.Series) -> pd.Series:
        return self.fit(X).transform(X)


class ContinuousDataEncoder:
    """Continuous variables encoder"""

    def __init__(
        self,
        n_components: int = 10,
        random_state: int = 0,
        weight_threshold: float = 0.005,
    ) -> None:
        self.model = BayesianGaussianMixture(
            n_components=n_components,
            random_state=random_state,
            weight_concentration_prior=1e-3,
        )
        self.n_components = n_components
        self.weight_threshold = weight_threshold
        self.weights: Optional[List[float]] = None
        self.std_multiplier = 4

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: pd.Series) -> Any:
        self.min_value = X.min()
        self.max_value = X.max()

        self.model.fit(X.values.reshape(-1, 1))
        self.weights = self.model.weights_

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def transform(self, X: pd.Series) -> pd.DataFrame:
        name = X.name
        X = X.values.reshape(-1, 1)
        means = self.model.means_.reshape(1, self.n_components)

        # predict cluster value
        stds = np.sqrt(self.model.covariances_).reshape(1, self.n_components)

        normalized_values = (X - means) / (self.std_multiplier * stds)

        # predict cluster
        component_probs = self.model.predict_proba(X)

        components = np.argmax(component_probs, axis=1)

        aranged = np.arange(len(X))
        normalized = normalized_values[aranged, components].reshape([-1, 1])
        normalized = np.clip(normalized, -0.99, 0.99).squeeze()

        out = np.stack([normalized, components], axis=1)

        return pd.DataFrame(out, columns=[f"{name}.value", f"{name}.component"])

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def inverse_transform(self, X: pd.DataFrame) -> pd.Series:
        normalized = np.clip(X.values[:, 0], -1, 1)
        means = self.model.means_.reshape([-1])
        stds = np.sqrt(self.model.covariances_).reshape([-1])
        selected_component = X.values[:, 1].astype(int)

        # recreate data
        std_t = stds[selected_component]
        mean_t = means[selected_component]
        reversed_data = normalized * self.std_multiplier * std_t + mean_t

        # clip values
        return np.clip(reversed_data, self.min_value, self.max_value)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform(self, X: pd.Series) -> pd.Series:
        return self.fit(X).transform(X)

    def components(self) -> int:
        if self.weights is None:
            raise RuntimeError("Train the model first")
        return len(self.weights)


class EncodersCallbacks:
    def __init__(self, encoders: dict) -> None:
        self.encoders = encoders

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for col in self.encoders:
            if col not in df.columns:
                continue
            enc = self.encoders[col]

            target = df[col]
            if hasattr(enc, "get_feature_names"):
                # onehot encoder
                encoded = pd.DataFrame(
                    enc.transform(target.values.reshape(-1, 1)),
                    columns=enc.get_feature_names([col]),
                    index=output.index.copy(),
                )
            else:
                # label encoder
                encoded = pd.DataFrame(
                    enc.transform(target),
                    columns=[col],
                    index=output.index.copy(),
                )

            orig_cols = list(output)
            old_col_idx = orig_cols.index(col)

            output.drop(columns=[col], inplace=True)
            l_cols, r_cols = output.columns[:old_col_idx], output.columns[old_col_idx:]

            out_cols = list(l_cols) + list(encoded.columns) + list(r_cols)
            output = pd.concat([output, encoded], axis=1)
            output = output[out_cols]

        return output

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        for col in self.encoders:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            if hasattr(enc, "get_feature_names"):
                columns = enc.get_feature_names([col])
            else:
                columns = [col]

            decoded = pd.DataFrame(
                enc.inverse_transform(output[columns].astype(int).values.squeeze()),
                columns=[col],
                index=output.index.copy(),
            )

            orig_cols = list(output.columns)
            col_inx = orig_cols.index(columns[0])

            output.drop(columns=columns, inplace=True)
            l_cols, r_cols = output.columns[:col_inx], output.columns[col_inx:]

            output = pd.concat([output, decoded], axis=1)
            out_cols = list(l_cols) + list(decoded.columns) + list(r_cols)
            output = output[out_cols]

        if output.isnull().values.any():
            raise RuntimeError("Imputation returned null")

        return output

    def numeric_decode(self, df: pd.DataFrame, strategy: str = "max") -> pd.DataFrame:
        output = df.copy()
        for col in self.encoders:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            if hasattr(enc, "get_feature_names"):
                columns = enc.get_feature_names([col])
            else:
                columns = [col]
            if strategy == "max":
                vals = output[columns].max(axis=1)
            else:
                raise ValueError(f"unknown strategy {strategy}")

            orig_cols = list(output.columns)
            col_inx = orig_cols.index(columns[0])

            output.drop(columns=columns, inplace=True)
            l_cols, r_cols = output.columns[:col_inx], output.columns[col_inx:]

            output[col] = vals

            out_cols = list(l_cols) + [col] + list(r_cols)
            output = output[out_cols]

        return output

    def __getitem__(self, key: str) -> Any:
        return self.encoders[key]


def dataframe_encode(data: pd.DataFrame) -> Tuple[pd.DataFrame, EncodersCallbacks]:
    encoders = {}
    encoded = data.copy()

    for col in encoded.columns:
        if (
            encoded[col].infer_objects().dtype.kind == "i"
            and encoded[col].min() == 0
            and encoded[col].max() == len(encoded[col].unique()) - 1
        ):
            continue

        if (
            encoded[col].infer_objects().dtype.kind in ["O", "b"]
            or len(encoded[col].unique()) < 15
        ):
            encoder = LabelEncoder().fit(encoded[col])
            encoded[col] = encoder.transform(encoded[col])
            encoders[col] = encoder
        elif encoded[col].infer_objects().dtype.kind in ["M"]:
            encoder = DatetimeEncoder().fit(encoded[col])
            encoded[col] = encoder.transform(encoded[col]).values
            encoders[col] = encoder

    return encoded, EncodersCallbacks(encoders)
