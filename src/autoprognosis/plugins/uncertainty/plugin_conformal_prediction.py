# stdlib
import copy
from typing import Any

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.uncertainty.base import UncertaintyPlugin
from autoprognosis.utils.risk_estimation import generate_dataset_for_horizon

percentile_val = 1.96


class ConformalPredictionPlugin(UncertaintyPlugin):
    """
    Uncertainty plugin based on the Conformal Prediction method.

    Args:
        model: model. The model to explain.
    """

    def __init__(
        self,
        model: Any,
        task_type: str = "classification",
        random_seed: int = 0,
    ) -> None:
        self.model = copy.deepcopy(model)
        self.random_seed = random_seed
        self.task_type = task_type

        self.calibration: dict = {}

    def _calibrate_classifier(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        y_pred = self.model.predict_proba(X)
        y_pred.index = y.index

        classes = y_pred.shape[1]

        for cls in range(classes):
            cls_filter = y == cls
            cls_preds = y_pred[cls_filter][cls]

            self.calibration[cls] = {
                "vals": cls_preds,
                "class_prob": cls_filter.sum() / len(y),
            }

    def _calibrate_regressor(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        y_pred = self.model.predict(X).values
        err = np.abs(y - y_pred)
        eps = np.quantile(err, 0.95)

        self.calibration = {
            "vals": eps,
        }

    def _calibrate_risk_estimation(
        self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame, time_horizons: list
    ) -> None:
        for horizon in time_horizons:
            X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
                X, T, Y, horizon
            )

            Y_horizon = np.asarray(Y_horizon)

            horizon_preds = self.model.predict(X_horizon, [horizon]).values.squeeze()

            safe_filter = Y_horizon == 0
            preds_safe = 1 - horizon_preds[safe_filter]
            preds_safe = np.asarray(sorted(preds_safe))

            risk_filter = Y_horizon == 1
            preds_risk = 1 - horizon_preds[risk_filter]
            preds_risk = np.asarray(sorted(preds_risk))

            # Compute FP/FN rate
            eps = 1e-8
            fpos = (preds_safe < 0.5).sum() + eps
            tpos = (preds_risk > 0.5).sum() + eps

            fneg = (preds_risk < 0.5).sum() + eps
            tneg = (preds_safe > 0.5).sum() + eps

            sensitivity = tpos / (tpos + fneg)
            specificity = tneg / (tneg + fpos)

            fn_rate = 1 - sensitivity
            fp_rate = 1 - specificity

            safe_prob = len(preds_safe) / len(horizon_preds)
            risk_prob = len(preds_risk) / len(horizon_preds)

            # Keep only valid examples
            preds_safe = preds_safe[preds_safe > 0.5]
            preds_risk = preds_risk[preds_risk > 0.5]

            self.calibration[horizon] = {
                "safe": preds_safe,
                "safe_prob": safe_prob,
                "risk": preds_risk,
                "risk_prob": risk_prob,
                "fp_rate": fp_rate,
                "fn_rate": fn_rate,
            }

    def _confidence_classifier(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.model.predict_proba(X).values
        confidence = []

        for pred in y_pred:
            cls = pred.argmax()
            prob = pred[cls]

            cls_cal = self.calibration[cls]["vals"]

            overconf = (prob > cls_cal).sum()
            sameconf = (prob == cls_cal).sum() + 1

            eps = (overconf + sameconf * np.random.uniform(0, 1)) / (len(cls_cal) + 1)

            confidence.append(eps)

        return np.asarray(confidence)

    def _confidence_regressor(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.calibration["vals"]] * len(X))

    def _confidence_risk_estimation(
        self, X: pd.DataFrame, time_horizons: list
    ) -> np.ndarray:
        confidence = []
        for horizon in time_horizons:
            predictions = self.model.predict(X, [horizon])
            predictions = np.asarray(predictions).squeeze()

            horizon_confidence = []

            for prediction in predictions:
                cohort_type = "risk"
                risk_lt_cal_cnt = (
                    prediction > self.calibration[horizon][cohort_type]
                ).sum()
                risk_eq_cal_cnt = (
                    prediction == self.calibration[horizon][cohort_type]
                ).sum() + 1
                risk_eps = (
                    risk_lt_cal_cnt + risk_eq_cal_cnt * np.random.uniform(0, 1)
                ) / (len(self.calibration[horizon][cohort_type]) + 1)

                safe_prediction = 1 - prediction
                cohort_type = "safe"
                safe_lt_cal_cnt = (
                    safe_prediction > self.calibration[horizon][cohort_type]
                ).sum()
                safe_eq_cal_cnt = (
                    safe_prediction == self.calibration[horizon][cohort_type]
                ).sum() + 1
                safe_eps = (
                    safe_lt_cal_cnt + safe_eq_cal_cnt * np.random.uniform(0, 1)
                ) / (len(self.calibration[horizon][cohort_type]) + 1)

                if prediction < 0.5:
                    horizon_confidence.append(
                        safe_eps * (1 - self.calibration[horizon]["fn_rate"])
                    )
                else:
                    horizon_confidence.append(
                        risk_eps * (1 - self.calibration[horizon]["fp_rate"])
                    )

            confidence.append(horizon_confidence)

        return np.asarray(confidence).T

    def fit(self, *args: Any, **kwargs: Any) -> "UncertaintyPlugin":
        if self.task_type == "classification":
            X = args[0]
            y = args[1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

            self.model.fit(X_train, y_train)

            self._calibrate_classifier(X_test, y_test)
        elif self.task_type == "regression":
            X = args[0]
            y = args[1]

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            self.model.fit(X_train, y_train)

            self._calibrate_regressor(X_test, y_test)
        elif self.task_type == "risk_estimation":
            X = args[0]
            T = args[1]
            y = args[2]

            eval_times = kwargs["time_horizons"]
            X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
                X, T, y
            )

            self.model.fit(X_train, T_train, y_train)

            self._calibrate_risk_estimation(X_test, T_test, y_test, eval_times)
        else:
            raise RuntimeError("task not supported", self.task_type)

        return self

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if self.task_type == "classification":
            mean = self.model.predict(*args)
            confidence = self._confidence_classifier(*args)
        elif self.task_type == "regression":
            mean = self.model.predict(*args)
            confidence = self._confidence_regressor(*args)
        elif self.task_type == "risk_estimation":
            eval_times = kwargs["time_horizons"]
            mean = self.model.predict(*args, eval_times)

            confidence = self._confidence_risk_estimation(*args, **kwargs)
        else:
            raise RuntimeError("task not supported", self.task_type)

        return mean.squeeze(), confidence.squeeze()

    def predict_proba(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if self.task_type == "classification":
            mean = self.model.predict_proba(*args)
            confidence = self._confidence_classifier(*args)
        else:
            raise RuntimeError("task not supported", self.task_type)

        return mean.squeeze(), confidence.squeeze()

    @staticmethod
    def name() -> str:
        return "conformal_prediction"


plugin = ConformalPredictionPlugin
