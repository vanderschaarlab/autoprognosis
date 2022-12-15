# stdlib
from typing import Any, Dict, List, Tuple, Type, Union

# third party
from optuna.trial import Trial

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_feature_selection_names,
)
import autoprognosis.logger as log
from autoprognosis.plugins.core.base_plugin import Plugin
import autoprognosis.plugins.core.params as params
from autoprognosis.plugins.imputers import Imputers
from autoprognosis.plugins.pipeline import Pipeline, PipelineMeta
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.plugins.preprocessors import Preprocessors

predefined_args = {
    "features_count": 10,
}


class PipelineSelector:
    """AutoML wrapper for pipelines

    Args:
        classifier: str
            Last estimator of the pipeline, the final classifier.
        calibration: int
            Type of calibration to use. 0 - none, 1 - sigmoid, 2 - isotonic.
        imputers: list
            list of imputers to sample from.
        feature_scaling: list
            list of feature scaling transformers to sample from.
        feature_selection: list
            list of feature selection methods ti sample from
        classifier_category: str
            task type: "classifier" or "risk_estimation"

    """

    def __init__(
        self,
        classifier: str,
        calibration: List[int] = [0, 1, 2],
        imputers: List[str] = [],
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        classifier_category: str = "classifier",  # "classifier", "risk_estimation", "regression"
    ) -> None:
        self.calibration = calibration
        self.imputers = [Imputers().get_type(plugin) for plugin in imputers]
        self.feature_scaling = [
            Preprocessors(category="feature_scaling").get_type(plugin)
            for plugin in feature_scaling
        ]
        self.feature_selection = [
            Preprocessors(category="dimensionality_reduction").get_type(plugin)
            for plugin in feature_selection
        ]

        if classifier == "multinomial_naive_bayes" or classifier == "bagging":
            self.feature_scaling = [
                Preprocessors(category="feature_scaling").get_type("minmax_scaler")
            ]
            self.feature_selection = []

        self.classifier = Predictions(category=classifier_category).get_type(classifier)

    def _generate_dist_name(self, key: str) -> str:
        if key == "imputation_candidate":
            imputers_str = [imputer.name() for imputer in self.imputers]
            imputers_str.sort()
            return f"{self.classifier.fqdn()}.imputation_candidate.{'_'.join(imputers_str)}"
        elif key == "feature_scaling_candidate":
            fs_str = [fs.name() for fs in self.feature_scaling]
            fs_str.sort()
            return (
                f"{self.classifier.fqdn()}.feature_scaling_candidate.{'_'.join(fs_str)}"
            )
        elif key == "feature_selection_candidate":
            fs_str = [fs.name() for fs in self.feature_selection]
            fs_str.sort()
            return f"{self.classifier.fqdn()}.feature_selection_candidate.{'_'.join(fs_str)}"
        else:
            raise ValueError(f"invalid key {key}")

    def hyperparameter_space(self) -> List:
        hp: List[Union[params.Integer, params.Categorical, params.Float]] = []

        if len(self.imputers) > 0:
            hp.append(
                params.Categorical(
                    self._generate_dist_name("imputation_candidate"),
                    [imputer.name() for imputer in self.imputers],
                )
            )
            for plugin in self.imputers:
                hp.extend(plugin.hyperparameter_space_fqdn(**predefined_args))

        if len(self.feature_scaling) > 0:
            hp.append(
                params.Categorical(
                    self._generate_dist_name("feature_scaling_candidate"),
                    [fs.name() for fs in self.feature_scaling],
                )
            )

        if len(self.feature_selection) > 0:
            hp.append(
                params.Categorical(
                    self._generate_dist_name("feature_selection_candidate"),
                    [fs.name() for fs in self.feature_selection],
                )
            )
            for plugin in self.feature_selection:
                hp.extend(plugin.hyperparameter_space_fqdn(**predefined_args))

        if len(self.calibration) > 0:
            hp.append(
                params.Integer(
                    self.classifier.fqdn() + ".calibration",
                    0,
                    len(self.calibration) - 1,
                ),
            )

        hp.extend(self.classifier.hyperparameter_space_fqdn())
        return hp

    def sample_hyperparameters(self, trial: Trial) -> Dict:
        params = self.hyperparameter_space()

        result = {}
        for param in params:
            result[param.name] = param.sample(trial)

        return result

    def sample_hyperparameters_np(self) -> Dict:
        params = self.hyperparameter_space()

        result = {}
        for param in params:
            result[param.name] = param.sample_np()

        return result

    def name(self) -> str:
        return self.classifier.name()

    def get_pipeline_template(
        self, search_domains: List[params.Params], hyperparams: List
    ) -> Tuple[List, Dict]:
        domain_list = [search_domains[k].name for k in range(len(search_domains))]

        model_list = list()

        calibration = hyperparams[
            domain_list.index(self.classifier.fqdn() + ".calibration")
        ]

        args: Dict[str, Dict] = {
            self.classifier.name(): {
                "calibration": int(calibration),
            },
        }

        def add_stage_hp(plugin: Type[Plugin]) -> None:
            if plugin.name() not in args:
                args[plugin.name()] = {}

            for param in plugin.hyperparameter_space_fqdn(**predefined_args):
                param_val = hyperparams[domain_list.index(param.name)]
                param_val = type(param.bounds[0])(param_val)

                args[plugin.name()][param.name.split(".")[-1]] = param_val

        if len(self.imputers) > 0:
            select_imp = hyperparams[
                domain_list.index(self._generate_dist_name("imputation_candidate"))
            ]
            selected = Imputers().get_type(select_imp)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        if len(self.feature_scaling) > 0:
            select_pre_fs = hyperparams[
                domain_list.index(self._generate_dist_name("feature_scaling_candidate"))
            ]
            selected = Preprocessors(category="feature_scaling").get_type(select_pre_fs)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        if len(self.feature_selection) > 0:
            select_pre_fs = hyperparams[
                domain_list.index(
                    self._generate_dist_name("feature_selection_candidate")
                )
            ]
            selected = Preprocessors(category="dimensionality_reduction").get_type(
                select_pre_fs
            )
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        # Add data cleanup
        cleaner = Preprocessors(category="dimensionality_reduction").get_type(
            "data_cleanup"
        )
        model_list.append(cleaner.fqdn())

        # Add predictor
        model_list.append(self.classifier.fqdn())
        add_stage_hp(self.classifier)

        log.info(f"[get_pipeline]: {model_list} -> {args}")

        return model_list, args

    def get_pipeline(
        self, search_domains: List[params.Params], hyperparams: List
    ) -> PipelineMeta:
        model_list, args = self.get_pipeline_template(search_domains, hyperparams)
        return self.get_pipeline_from_template(model_list, args)

    def get_pipeline_from_template(self, model_list: List, args: Dict) -> PipelineMeta:
        return Pipeline(model_list)(args)

    def get_pipeline_from_named_args(self, **kwargs: Any) -> PipelineMeta:
        model_list = list()

        pipeline_args: dict = {}

        def add_stage_hp(plugin: Type[Plugin]) -> None:
            if plugin.name() not in pipeline_args:
                pipeline_args[plugin.name()] = {}

            for param in plugin.hyperparameter_space_fqdn(**predefined_args):
                if param.name not in kwargs:
                    continue
                param_val = kwargs[param.name]
                param_val = type(param.bounds[0])(param_val)

                pipeline_args[plugin.name()][param.name.split(".")[-1]] = param_val

        imputation_key = self._generate_dist_name("imputation_candidate")

        if imputation_key in kwargs:
            idx = kwargs[imputation_key]
            selected = Imputers().get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)
        elif len(self.imputers) > 0:
            model_list.append(self.imputers[0].fqdn())
            add_stage_hp(self.imputers[0])

        pre_key = self._generate_dist_name("feature_selection_candidate")
        if pre_key in kwargs:
            idx = kwargs[pre_key]
            selected = Preprocessors(category="dimensionality_reduction").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        pre_key = self._generate_dist_name("feature_scaling_candidate")
        if pre_key in kwargs:
            idx = kwargs[pre_key]
            selected = Preprocessors(category="feature_scaling").get_type(idx)
            model_list.append(selected.fqdn())
            add_stage_hp(selected)

        # Add data cleanup
        cleaner = Preprocessors(category="dimensionality_reduction").get_type(
            "data_cleanup"
        )
        model_list.append(cleaner.fqdn())

        # Add predictor
        model_list.append(self.classifier.fqdn())
        add_stage_hp(self.classifier)

        return Pipeline(model_list)(pipeline_args)
