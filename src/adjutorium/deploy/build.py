# stdlib
import atexit
import json
from pathlib import Path
import queue
import threading
import traceback
from typing import Any, Optional, Tuple, Union

# third party
import pandas as pd

# adjutorium absolute
from adjutorium.deploy.proto import NewClassificationAppProto, NewRiskEstimationAppProto
from adjutorium.deploy.utils import file_copy, file_md5
from adjutorium.exceptions import BuildCancelled
import adjutorium.logger as log
from adjutorium.plugins.prediction import Predictions
from adjutorium.studies._preprocessing import dataframe_encode_and_impute
from adjutorium.utils.serialization import load_model_from_file, save_model_to_file

STATUS_KEY = "build_status"
CHECKPOINT_KEY = "checkpoint"

STATUS_DONE = "done"
STATUS_CANCELLED = "cancelled"
STATUS_STARTED = "running"
STATUS_PENDING = "pending"

CHECKPOINT_STARTED = "started"
CHECKPOINT_DATASET_LOADING = "dataset_loading"
CHECKPOINT_DATASET_ENCODING = "dataset_encoding"
CHECKPOINT_DATASET_IMPUTATION = "dataset_imputation"
CHECKPOINT_TRAIN_MODEL = "train_model"
CHECKPOINT_DONE = "done"
CHECKPOINT_FAILED = "failed"

TASK_TTL = 5 * 60

DISPLAY_NAME = "Adjutorium model"

q: Any = queue.Queue()


def worker() -> None:
    while True:
        cbk, app_path, done_cbk = q.get()
        log.info(f"Working on {app_path}")

        try:
            cbk(app_path)
        except BaseException:
            traceback.print_exc()
        finally:
            done_cbk()

        q.task_done()


threading.Thread(target=worker, daemon=True).start()


class BuilderProgress:
    def __init__(self, app_id: str) -> None:
        self.key = app_id
        self.cache: dict = {}

    def cancel(self) -> None:
        self.status = STATUS_CANCELLED

    def reset(self) -> None:
        self.cache = {}

    @property
    def status(self) -> Optional[str]:
        return self.cache.get(STATUS_KEY)

    @status.setter
    def status(self, state: str) -> None:
        self.cache[STATUS_KEY] = state

    @property
    def checkpoint(self) -> Optional[str]:
        return self.cache.get(CHECKPOINT_KEY)

    @checkpoint.setter
    def checkpoint(self, state: str) -> None:
        prev = []
        if state != STATUS_DONE and state != STATUS_STARTED:
            cached = self.cache.get(CHECKPOINT_KEY)
            if cached is not None:
                try:
                    prev = json.loads(cached)
                except BaseException:
                    prev = []

            prev.append(state)
        self.cache[CHECKPOINT_KEY] = json.dumps(prev)


class Builder:
    def __init__(
        self,
        task: Union[NewRiskEstimationAppProto, NewClassificationAppProto],
        blocking: bool = True,
        use_cache: bool = False,
        comparative_models: list = [],
    ) -> None:
        self.task = task
        self.use_cache = use_cache

        self.blocking = blocking
        self.model_path = Path(self.task.model_path)
        self.working_path = self.model_path.parent
        self.comparative_models = comparative_models

        self.trained_model_path = self.working_path / "trained_models.p"
        self.app_backup_file = self.working_path / "app.p"

        self.progress = BuilderProgress(str(self.app_backup_file))
        atexit.register(self.progress.reset)

    def _load_dataset(self) -> Tuple:
        self._should_continue()
        self.checkpoint = CHECKPOINT_DATASET_LOADING

        data = pd.read_csv(self.task.dataset_path)
        imputation_method: Optional[str] = None
        # we treat binary columns as checkboxes
        checkboxes: list = []
        other_cols: list = []

        if self.task.type == "risk_estimation":
            data = data[data[self.task.time_column] > 0]

            X = data.drop(
                columns=[
                    self.task.time_column,
                    self.task.target_column,
                ]
            )
            T = data[self.task.time_column]
            Y = data[self.task.target_column]
            log.info(f"Loaded dataset {X.shape} {T.shape} {Y.shape}")

            self._should_continue()

            self.checkpoint = CHECKPOINT_DATASET_IMPUTATION
            self.checkpoint = CHECKPOINT_DATASET_ENCODING

            if len(self.task.imputers) == 1:
                imputation_method = self.task.imputers[0]

            rawX = X.copy()

            X, encoders = dataframe_encode_and_impute(
                X, imputation_method=imputation_method
            )
            log.info(f"Loaded dataset after encoding {X.shape} {T.shape} {Y.shape}")

            for col in rawX.columns:
                vals = [v for v in rawX[col].unique() if not pd.isna(v)]
                log.info(f"unique vals {vals}")
                if sorted(vals) == [0, 1]:
                    checkboxes.append(col)
                else:
                    other_cols.append(col)

            rawX = rawX[checkboxes + other_cols]
            X = encoders.encode(rawX)
            rawX = encoders.decode(X.dropna())
            log.info(f"Loaded dataset final encoding {X.shape} {T.shape} {Y.shape}")

            return (
                X,
                rawX,
                T,
                Y,
                encoders,
                checkboxes,
            )
        elif self.task.type == "classification":
            X = data.drop(
                columns=[
                    self.task.target_column,
                ]
            )
            Y = data[self.task.target_column]
            log.info(f"Loaded dataset {X.shape} {Y.shape}")

            self._should_continue()

            self.checkpoint = CHECKPOINT_DATASET_IMPUTATION
            self.checkpoint = CHECKPOINT_DATASET_ENCODING

            if len(self.task.imputers) == 1:
                imputation_method = self.task.imputers[0]

            rawX = X.copy()

            X, encoders = dataframe_encode_and_impute(
                X, imputation_method=imputation_method
            )
            log.info(f"Loaded dataset after encoding {X.shape} {Y.shape}")

            for col in rawX.columns:
                vals = [v for v in rawX[col].unique() if not pd.isna(v)]
                log.info(f"unique vals {vals}")
                if sorted(vals) == [0, 1]:
                    checkboxes.append(col)
                else:
                    other_cols.append(col)

            rawX = rawX[checkboxes + other_cols]
            X = encoders.encode(rawX)
            rawX = encoders.decode(X.dropna())
            log.info(f"Loaded dataset final encoding {X.shape} {Y.shape}")

            return (
                X,
                rawX,
                None,
                Y,
                encoders,
                checkboxes,
            )
        raise NotImplementedError(f"task not supported {self.task.type}")

    def _load_models_risk_estimation(
        self,
        X: pd.DataFrame,
        T: pd.DataFrame,
        Y: pd.DataFrame,
        time_horizons: list,
        output_path: Path,
        explainers: list,
    ) -> dict:
        self._should_continue()
        log.info(f"Creating model with explainers {explainers}")
        model = load_model_from_file(self.task.model_path)
        model.enable_explainer(explainer_plugins=explainers, explanations_nepoch=500)
        log.info(f"Loaded model {model.name()}")

        self.checkpoint = CHECKPOINT_TRAIN_MODEL
        self._should_continue()
        model.fit(X, T, Y)

        self._should_continue()
        app_models = {
            DISPLAY_NAME: model,
        }

        plugins = Predictions(category="risk_estimation")
        for name, comparative in self.comparative_models:
            ref_model = plugins.get(comparative)
            ref_model.fit(X, T, Y)

            app_models[name] = ref_model

        self._should_continue()
        save_model_to_file(output_path, app_models)

        return app_models

    def _load_models_classification(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        output_path: Path,
        explainers: list,
    ) -> dict:
        self._should_continue()
        log.info(f"Creating model with explainers {explainers}")
        model = load_model_from_file(self.task.model_path)
        model.enable_explainer(explainer_plugins=explainers, explanations_nepoch=500)
        log.info(f"Loaded model {model.name()}")

        self.checkpoint = CHECKPOINT_TRAIN_MODEL
        self._should_continue()
        model.fit(X, Y)

        self._should_continue()
        app_models = {
            DISPLAY_NAME: model,
        }

        self._should_continue()
        save_model_to_file(output_path, app_models)

        return app_models

    def _run(self, app_path: Path) -> str:
        self._should_continue()
        X, rawX, T, Y, encoders, checkboxes = self._load_dataset()

        self._should_continue()
        if self.task.type == "risk_estimation":
            app_models = self._load_models_risk_estimation(
                X,
                T,
                Y,
                self.task.horizons,
                output_path=self.trained_model_path,
                explainers=self.task.explainers,
            )
        elif self.task.type == "classification":
            app_models = self._load_models_classification(
                X,
                Y,
                output_path=self.trained_model_path,
                explainers=self.task.explainers,
            )

        else:
            raise RuntimeError(f"invalid task type {self.task.type}")

        self._should_continue()

        app_title = self.task.name
        banner_title = f"{app_title} study"

        if self.task.dashboard_type == "streamlit":
            # adjutorium absolute
            from adjutorium.apps.common.pandas_to_streamlit import (
                generate_menu as generate_menu_with_streamlit,
            )

            column_types = generate_menu_with_streamlit(rawX, checkboxes)
            menu_components = column_types
        elif self.task.dashboard_type == "dash":
            # adjutorium absolute
            from adjutorium.apps.common.pandas_to_dash import (
                generate_menu as generate_menu_with_dash,
            )

            menu_components, column_types = generate_menu_with_dash(rawX, checkboxes)
        else:
            raise RuntimeError("invalid dashboard type", self.task.dashboard_type)

        plot_alternatives: dict = {"Adjutorium model": {}}
        for col in self.task.plot_alternatives:
            if col not in rawX.columns:
                raise ValueError(f"Invalid col for split {col} {X.columns}")
            plot_alternatives[DISPLAY_NAME][col] = list(rawX[col].unique())

        if self.task.type == "risk_estimation":
            save_model_to_file(
                app_path,
                {
                    "title": app_title,
                    "type": self.task.type,
                    "banner_title": banner_title,
                    "models": app_models,
                    "column_types": column_types,
                    "encoders": encoders,
                    "menu_components": menu_components,
                    "time_horizons": self.task.horizons,
                    "plot_alternatives": plot_alternatives,
                },
            )
        elif self.task.type == "classification":
            save_model_to_file(
                app_path,
                {
                    "title": app_title,
                    "type": self.task.type,
                    "banner_title": banner_title,
                    "models": app_models,
                    "column_types": column_types,
                    "encoders": encoders,
                    "menu_components": menu_components,
                    "plot_alternatives": plot_alternatives,
                },
            )
        file_copy(app_path, self.app_backup_file)
        self.checkpoint = CHECKPOINT_DONE

        return str(self.app_backup_file)

    def run(self) -> str:
        model_version = file_md5(Path(self.task.model_path))
        log.info(f"model_hash {model_version}")

        app_build_file = self.working_path / f"app_{model_version}.p"

        if app_build_file.exists() and self.use_cache:
            self.status = STATUS_DONE
            self.checkpoint = STATUS_DONE

            log.info(f"app already built for this model {app_build_file}")
            file_copy(app_build_file, self.app_backup_file)
            return str(self.app_backup_file)

        if self.status == STATUS_STARTED:
            log.info(f"build already running {self.task}")
            return str(self.app_backup_file)

        self.status = STATUS_STARTED
        self.checkpoint = STATUS_STARTED

        self._should_continue()

        def done_cbk() -> None:
            self.status = STATUS_DONE
            self.progress.reset()
            log.info("Building done")

        if self.blocking:
            self._run(app_build_file)
        else:
            q.put((self._run, app_build_file, done_cbk))

        return str(self.app_backup_file)

    def _should_continue(self) -> None:
        if self.status == STATUS_CANCELLED:
            raise BuildCancelled("the build was cancelled")

    def cancel(self) -> None:
        self.status = STATUS_CANCELLED

    @property
    def status(self) -> Optional[str]:
        return self.progress.status

    @status.setter
    def status(self, state: str) -> None:
        self.progress.status = state

    @property
    def checkpoint(self) -> Optional[str]:
        return self.progress.checkpoint

    @checkpoint.setter
    def checkpoint(self, state: str) -> None:
        self.progress.checkpoint = state
