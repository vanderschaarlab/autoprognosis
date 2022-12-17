# stdlib
import copy
from typing import Any, Dict, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# autoprognosis absolute
from autoprognosis.plugins.explainers import Explainers
from autoprognosis.plugins.uncertainty.base import UncertaintyPlugin
from autoprognosis.utils.risk_estimation import generate_dataset_for_horizon


class CohortRule:
    def __init__(
        self,
        name: str,
        target_features: list = [],
        ops: list = [],
        thresholds: list = [],
    ) -> None:
        self.name = name
        self.target_features = target_features
        self.ops = ops
        self.thresholds = thresholds

        self.calibration_perf_score: float = 1
        self.calibration_predictions_safe: np.ndarray = []
        self.calibration_predictions_risk: np.ndarray = []
        self.calibration_fn_rate: float = 1
        self.calibration_fp_rate: float = 1

        self._check_sanity()

    def calibrate(self, ground_truth: np.ndarray, predictions: np.ndarray) -> bool:
        if len(predictions) < 2:
            return False

        calibration_predictions = np.asarray(predictions).squeeze()
        calibration_truth = np.asarray(ground_truth).squeeze()

        try:
            self.calibration_perf_score = roc_auc_score(
                calibration_truth, calibration_predictions
            )
        except BaseException:
            self.calibration_perf_score = 1e-8

        safe_filter = calibration_truth == 0
        self.calibration_predictions_safe = 1 - calibration_predictions[safe_filter]
        self.calibration_predictions_safe = np.asarray(
            sorted(self.calibration_predictions_safe)
        )

        risk_filter = calibration_truth == 1
        self.calibration_predictions_risk = calibration_predictions[risk_filter]
        self.calibration_predictions_risk = np.asarray(
            sorted(self.calibration_predictions_risk)
        )

        # Compute FP/FN rate
        eps = 1e-8
        fpos = (self.calibration_predictions_safe < 0.5).sum() + eps
        tpos = (self.calibration_predictions_risk > 0.5).sum() + eps

        fneg = (self.calibration_predictions_risk < 0.5).sum() + eps
        tneg = (self.calibration_predictions_safe > 0.5).sum() + eps

        sensitivity = tpos / (tpos + fneg)
        specificity = tneg / (tneg + fpos)

        self.calibration_fn_rate = 1 - sensitivity
        self.calibration_fp_rate = 1 - specificity

        self.calibration_safe_prob = (
            len(self.calibration_predictions_safe) / len(calibration_predictions) + eps
        )
        self.calibration_risk_prob = (
            len(self.calibration_predictions_risk) / len(calibration_predictions) + eps
        )

        # Keep only valid examples
        self.calibration_predictions_safe = self.calibration_predictions_safe[
            self.calibration_predictions_safe > 0.5
        ]
        self.calibration_predictions_risk = self.calibration_predictions_risk[
            self.calibration_predictions_risk > 0.5
        ]
        return (
            len(self.calibration_predictions_safe)
            + len(self.calibration_predictions_risk)
            > 0
        )

    def get_confidence(self, prediction: np.ndarray) -> np.ndarray:
        if not (
            len(self.calibration_predictions_safe)
            + len(self.calibration_predictions_risk)
            > 0
        ):
            raise RuntimeError(f"Uncalibrated cohort {self.name}")

        prediction = np.asarray(prediction).squeeze()

        risk_lt_cal_cnt = (prediction > self.calibration_predictions_risk).sum()
        risk_eq_cal_cnt = (prediction == self.calibration_predictions_risk).sum() + 1
        risk_eps = (risk_lt_cal_cnt + risk_eq_cal_cnt * np.random.uniform(0, 1)) / (
            len(self.calibration_predictions_risk) + 1
        )

        safe_prediction = 1 - prediction
        safe_lt_cal_cnt = (safe_prediction > self.calibration_predictions_safe).sum()
        safe_eq_cal_cnt = (
            safe_prediction == self.calibration_predictions_safe
        ).sum() + 1
        safe_eps = (safe_lt_cal_cnt + safe_eq_cal_cnt * np.random.uniform(0, 1)) / (
            len(self.calibration_predictions_safe) + 1
        )

        if prediction < 0.5:
            return safe_eps * (1 - self.calibration_fn_rate)
        else:
            return risk_eps * (1 - self.calibration_fp_rate)

    def get_difficulty(self) -> float:
        return round(1 - self.calibration_perf_score, 2)

    def _check_sanity(self) -> None:
        if len(self.target_features) != len(self.ops):
            raise ValueError("invalid self.ops")
        if len(self.target_features) != len(self.thresholds):
            raise ValueError("invalid self.thresholds")

    def _eval(self, X: pd.DataFrame, feature: str, op: str, threshold: Any) -> bool:
        if op == "lt":
            return X[feature] < threshold
        elif op == "le":
            return X[feature] <= threshold
        elif op == "gt":
            return X[feature] > threshold
        elif op == "ge":
            return X[feature] >= threshold
        elif op == "eq":
            return X[feature] == threshold
        else:
            raise RuntimeError("unsupported operation", op)

    def match(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        res = pd.Series([True] * len(X), index=X.index)
        for idx in range(len(self.target_features)):
            res &= self._eval(
                X, self.target_features[idx], self.ops[idx], self.thresholds[idx]
            )
        return res

    def merge(self, other: "CohortRule") -> "CohortRule":
        self.name = f"{self.name} & {other.name}"
        self.target_features += other.target_features
        self.ops += other.ops
        self.thresholds += other.thresholds

        self._check_sanity()

        return self


class CohortMgmt:
    global_cohort = "(*)"

    def __init__(self, cohort_rules: Dict[str, CohortRule]) -> None:
        if not (CohortMgmt.global_cohort in cohort_rules):
            raise ValueError("Provide a rule for the full population")

        self.cohort_scores = sorted(
            cohort_rules, key=lambda key: cohort_rules[key].calibration_perf_score
        )
        self.cohort_rules = cohort_rules

    def match(self, X: pd.DataFrame) -> Optional[CohortRule]:
        for rule_name in self.cohort_scores:
            rule = self.cohort_rules[rule_name]

            if rule.match(X).sum() != 0:
                return rule

        return None

    def match_all(self, X: pd.DataFrame) -> List[CohortRule]:
        results = []
        for rule_name in self.cohort_scores:
            rule = self.cohort_rules[rule_name]
            if rule.match(X).sum() != 0:
                results.append(rule)

        return results

    def diagnostics_headers(self) -> list:
        return [
            "global_confidence",
            "avg_confidence",
            "high_fn_rate",
            "high_fp_rate",
            "high_imbalance",
        ]

    def diagnostics(
        self, X: pd.DataFrame, prediction: pd.DataFrame, cohort_limit: int = 2
    ) -> dict:
        _rules = self.match_all(X)

        all_conf = []
        global_conf = 0
        high_fn_rate = []
        high_fp_rate = []
        high_imbalance = []

        for rule in _rules:
            conf = rule.get_confidence(prediction)
            all_conf.append(conf)
            if rule.name == CohortMgmt.global_cohort:
                global_conf = conf
            if rule.calibration_fn_rate >= 0.5:
                high_fn_rate.append(rule.name)
            if rule.calibration_fp_rate >= 0.5:
                high_fp_rate.append(rule.name)
            if rule.calibration_safe_prob >= 0.8 or rule.calibration_risk_prob >= 0.8:
                high_imbalance.append(rule.name)

        return {
            "global_confidence": global_conf,
            "avg_confidence": np.mean(all_conf),
            "high_fn_rate": high_fn_rate[-cohort_limit:],
            "high_fp_rate": high_fp_rate[-cohort_limit:],
            "high_imbalance": high_imbalance[-cohort_limit:],
        }

    def diagnostics_df(self, X: pd.DataFrame, prediction: pd.DataFrame) -> dict:
        diags = self.diagnostics(X, prediction)
        return pd.DataFrame(
            [
                diags.values(),
            ],
            columns=diags.keys(),
        )

    def get(self, name: str) -> Optional[CohortRule]:
        return self.cohort_rules[name]


class CohortExplainerPlugin(UncertaintyPlugin):
    """
    Uncertainty plugin based on Conformal Prediction and cohort analysis.

    Args:
        model: model.
            The model to explain.
        task_type: str
            risk_estimation, regression, classification.
        random_seed: int
            Random seed
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

        self.cohort_calibration: Dict[float, CohortMgmt] = {}

    def _sample_calibration_filter(self, _filter: pd.Index) -> pd.Index:
        _selected_size = int(0.2 * _filter.sum())

        _selected_rows = _filter.index[_filter == True].tolist()  # noqa

        _discarded_rows = _selected_rows[_selected_size:]
        _filter.iloc[_discarded_rows] = False

        return _filter

    def _sample_calibration_set(
        self, X: pd.DataFrame, Y: pd.DataFrame, _filter: bool = True
    ) -> Tuple[pd.Index, pd.Index]:
        cens_filter = self._sample_calibration_filter((Y == 0) & _filter)
        ev_filter = self._sample_calibration_filter((Y == 1) & _filter)

        return cens_filter, ev_filter

    def _discover_cohorts_by_performance(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        cohort_splits: list,
        prev_rule: Optional[CohortRule] = None,
    ) -> list:
        if len(cohort_splits) == 0:
            return []

        calibration_filters = []

        for idx, feature in enumerate(cohort_splits):
            if feature not in X.columns:
                continue

            values = sorted(X[feature].unique())
            eval_filters = []

            if len(values) == 1:
                continue

            if len(values) < 5:
                for v in values:
                    lfilter_str = f"({feature} == {v})"
                    eval_filters.append(CohortRule(lfilter_str, [feature], ["eq"], [v]))
            else:
                median = round(np.median(values), 2)
                lfilter_str = f"({feature} < {median})"
                rfilter_str = f"({feature} >= {median})"
                eval_filters = [
                    CohortRule(lfilter_str, [feature], ["lt"], [median]),
                    CohortRule(rfilter_str, [feature], ["ge"], [median]),
                ]

            for rule in eval_filters:
                eval_rule = rule
                if prev_rule is not None:
                    eval_rule.merge(prev_rule)

                local_filter = eval_rule.match(X)

                if local_filter.sum() == 0:
                    continue

                cens_filter, ev_filter = self._sample_calibration_set(
                    X, Y, local_filter
                )

                if len(cens_filter) == 0 or len(ev_filter) == 0:
                    continue

                calibration_filters.append((eval_rule, cens_filter | ev_filter))

                if prev_rule is None:
                    ext_cal_filters = self._discover_cohorts_by_performance(
                        X, Y, cohort_splits[idx + 1 :], eval_rule
                    )
                    calibration_filters.extend(ext_cal_filters)

        return calibration_filters

    def _calibrate_classifier(self, X_eval: pd.DataFrame, Y_eval: pd.DataFrame) -> None:
        # Learn baseline feature importance
        X_eval = pd.DataFrame(X_eval)
        Y_eval = pd.Series(Y_eval, index=X_eval.index)

        pre_model = copy.deepcopy(self.model)
        pre_model.fit(X_eval, Y_eval)

        exp = Explainers().get(
            "risk_effect_size",
            pre_model,
            X_eval,
            Y_eval,
            task_type="classification",
            prefit=True,
            effect_size=0.5,
        )
        important_cols = exp.explain(X_eval).index.tolist()

        # Prepare cohort map based on feature importance
        calibration_filters = self._discover_cohorts_by_performance(
            X_eval, Y_eval, important_cols
        )

        all_calibration_filters = calibration_filters[0][1]

        for filter_rule, fl in calibration_filters:
            all_calibration_filters |= fl

        training_filter = ~all_calibration_filters

        # Train the baseline model
        self.model = copy.deepcopy(self.model)
        self.model.fit(X_eval[training_filter], Y_eval[training_filter])

        # Evaluate the calibration cohorts
        calibration_scores = {}

        def _eval_cohort(
            eval_filter: pd.Index,
            filter_rule: CohortRule,
            X: pd.DataFrame,
            Y: pd.DataFrame,
        ) -> None:
            nonlocal calibration_scores
            X_test = X[eval_filter]
            Y_test = Y[eval_filter]

            if len(X_test) < 2:
                return

            Y_preds = self.model.predict_proba(X_test)
            Y_preds = np.asarray(Y_preds)[:, 1]

            calibrated = filter_rule.calibrate(
                np.asarray(Y_test).squeeze(), Y_preds.squeeze()
            )

            if not calibrated:
                return

            calibration_scores[filter_rule.name] = filter_rule

        for filter_rule, eval_filter in calibration_filters:
            _eval_cohort(eval_filter, filter_rule, X_eval, Y_eval)

        # Sample full population
        cens_filter, ev_filter = self._sample_calibration_set(X_eval, Y_eval)
        eval_filter = cens_filter | ev_filter
        filter_rule = CohortRule(CohortMgmt.global_cohort)
        _eval_cohort(eval_filter, filter_rule, X_eval, Y_eval)

        # Return Cohort Mgmt
        self.cohort_calibration[0] = CohortMgmt(calibration_scores)

    def _calibrate_risk_estimation(
        self,
        X_eval: pd.DataFrame,
        T_eval: pd.DataFrame,
        Y_eval: pd.DataFrame,
        target_horizons: list,
    ) -> None:
        self.cohort_calibration = {}
        for target_horizon in target_horizons:
            X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
                X_eval, T_eval, Y_eval, target_horizon
            )

            # Learn baseline feature importance
            pre_model = copy.deepcopy(self.model)
            pre_model.fit(X_horizon, T_horizon, Y_horizon)

            exp = Explainers().get(
                "risk_effect_size",
                pre_model,
                X_horizon,
                Y_horizon,
                task_type="risk_estimation",
                prefit=True,
                effect_size=0.5,
                time_to_event=T_horizon,
                eval_times=[target_horizon],
            )
            important_cols = exp.explain(X_horizon).index.tolist()

            # Prepare cohort map based on feature importance
            calibration_filters = self._discover_cohorts_by_performance(
                X_horizon, Y_horizon, important_cols
            )

            all_calibration_filters = calibration_filters[0][1]

            for filter_rule, fl in calibration_filters:
                all_calibration_filters |= fl

            training_filter = ~all_calibration_filters

            # Train the baseline model
            self.model.fit(
                X_horizon[training_filter],
                T_horizon[training_filter],
                Y_horizon[training_filter],
            )

            # Evaluate the calibration cohorts
            calibration_scores = {}

            def _eval_cohort(
                eval_filter: pd.Index,
                filter_rule: CohortRule,
                X: pd.DataFrame,
                Y: pd.DataFrame,
                target_horizon: int,
            ) -> None:
                nonlocal calibration_scores
                X_test = X[eval_filter]
                Y_test = Y[eval_filter]

                if len(X_test) < 2:
                    return

                Y_preds = self.model.predict(X_test, [target_horizon])

                calibrated = filter_rule.calibrate(
                    Y_test.values.squeeze(), Y_preds.values.squeeze()
                )

                if not calibrated:
                    return

                calibration_scores[filter_rule.name] = filter_rule

            for filter_rule, eval_filter in calibration_filters:
                _eval_cohort(
                    eval_filter, filter_rule, X_horizon, Y_horizon, target_horizon
                )

            # Sample full population
            cens_filter, ev_filter = self._sample_calibration_set(X_horizon, Y_horizon)
            eval_filter = cens_filter | ev_filter
            filter_rule = CohortRule(CohortMgmt.global_cohort)
            _eval_cohort(
                eval_filter,
                filter_rule,
                X_horizon,
                Y_horizon,
                target_horizon,
            )

            # Return Cohort Mgmt
            self.cohort_calibration[target_horizon] = CohortMgmt(calibration_scores)

    def _confidence_classifier(self, X: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(
            [], columns=self.cohort_calibration[0].diagnostics_headers()
        )
        for idx, row in X.iterrows():
            eval_data = X.loc[[idx]]

            y_pred = self.model.predict_proba(eval_data)
            y_pred = np.asarray(y_pred)[:, 1]

            diags = self.cohort_calibration[0].diagnostics_df(eval_data, y_pred)
            output = output.append(diags)

        return output

    def _confidence_risk_estimation(
        self, X: pd.DataFrame, time_horizons: list
    ) -> pd.DataFrame:
        output = pd.DataFrame(
            [],
            columns=self.cohort_calibration[time_horizons[0]].diagnostics_headers()
            + ["horizon"],
        )
        for horizon in time_horizons:
            for idx, row in X.iterrows():
                eval_data = X.loc[[idx]]

                y_pred = self.model.predict(eval_data, [horizon])
                y_pred = np.asarray(y_pred).squeeze()

                diags = self.cohort_calibration[horizon].diagnostics_df(
                    eval_data, y_pred
                )
                diags["horizon"] = horizon
                output = output.append(diags)

        return output

    def fit(self, *args: Any, **kwargs: Any) -> "UncertaintyPlugin":
        if self.task_type == "classification":
            X = args[0]
            y = args[1]

            self._calibrate_classifier(X, y)
        elif self.task_type == "risk_estimation":
            X = args[0]
            T = args[1]
            y = args[2]

            eval_times = kwargs["time_horizons"]
            self._calibrate_risk_estimation(X, T, y, eval_times)
        else:
            raise RuntimeError("task not supported", self.task_type)

        return self

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = pd.DataFrame(args[0])
        if self.task_type == "classification":
            mean = self.model.predict(X)
            confidence = self._confidence_classifier(X)
        elif self.task_type == "risk_estimation":
            eval_times = kwargs["time_horizons"]
            mean = self.model.predict(X, eval_times)

            confidence = self._confidence_risk_estimation(X, eval_times)
        else:
            raise RuntimeError("task not supported", self.task_type)

        return mean.squeeze(), confidence

    def predict_proba(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = pd.DataFrame(args[0])
        if self.task_type == "classification":
            mean = self.model.predict_proba(X)
            confidence = self._confidence_classifier(X)
        else:
            raise RuntimeError("task not supported", self.task_type)

        return mean.squeeze(), confidence

    @staticmethod
    def name() -> str:
        return "cohort_explainer"


plugin = CohortExplainerPlugin
