# Adapted from https://github.com/yzhao062/combo/blob/master/combo/models/classifier_stacking.py
"""Stacking (meta ensembling). See http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
for more information.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

# stdlib
from abc import ABC, abstractmethod
from collections import defaultdict
import copy
from inspect import signature
import warnings

# third party
from numpy import percentile
import numpy as np
import pandas as pd
from pyod.utils.utility import check_parameter
from scipy.special import erf
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import (
    check_array,
    check_random_state,
    check_X_y,
    column_or_1d,
    shuffle,
)
from sklearn.utils.extmath import weighted_mode
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class BaseAggregator(ABC):
    """Abstract class for all combination classes.

        Stacking (meta ensembling). See http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/for more information.


    Parameters
    ----------
    base_estimators: list, length must be greater than 1
        A list of base estimators. Certain methods must be present, e.g.,
        `fit` and `predict`.

    pre_fitted: bool, optional (default=False)
        Whether the base estimators are trained. If True, `fit`
        process may be skipped.
    """

    @abstractmethod
    def __init__(self, base_estimators, pre_fitted=False):
        if not isinstance(base_estimators, (list)):
            raise ValueError("Invalid base_estimators")

        if len(base_estimators) < 2:
            raise ValueError("At least 2 estimators are required")
        self.base_estimators = base_estimators
        self.n_base_estimators_ = len(self.base_estimators)
        self.pre_fitted = pre_fitted

    @abstractmethod
    def fit(self, X, y=None):
        """Fit estimator. y is optional for unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).

        Returns
        -------
        self
        """
        pass

    # todo: make sure fit then predict is equivalent to fit_predict
    @abstractmethod
    def fit_predict(self, X, y=None):
        """Fit estimator and predict on X. y is optional for unsupervised
        methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : numpy array of shape (n_samples,)
            The class probabilities of the input samples.
            Classes are ordered by lexicographic order.
        """
        pass

    def _process_decision_scores(self):
        """Internal function to calculate key attributes for outlier detection
        combination tasks.

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """

        self.threshold_ = percentile(
            self.decision_scores_, 100 * (1 - self.contamination)
        )
        self.labels_ = (self.decision_scores_ > self.threshold_).astype("int").ravel()

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    def _detector_predict(self, X):
        """Internal function to predict if a particular sample is an
        outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        check_is_fitted(self, ["decision_scores_", "threshold_", "labels_"])

        pred_score = self.decision_function(X)
        return (pred_score > self.threshold_).astype("int").ravel()

    def _detector_predict_proba(self, X, proba_method="linear"):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:

        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        proba_method : str, optional (default='linear')
            Probability conversion method. It must be one of
            'linear' or 'unify'.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1].
        """

        check_is_fitted(self, ["decision_scores_", "threshold_", "labels_"])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(X)

        probs = np.zeros([X.shape[0], int(self._classes)])
        if proba_method == "linear":
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = (
                scaler.transform(test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            )
            probs[:, 0] = 1 - probs[:, 1]
            return probs

        elif proba_method == "unify":
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (self._sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(
                proba_method, "is not a valid probability conversion method"
            )

    def _set_n_classes(self, y):
        """Set the number of classes if `y` is presented.

        Parameters
        ----------
        y : numpy array of shape (n_samples,)
            Ground truth.

        Returns
        -------
        self
        """

        self._classes = 2  # default as binary classification
        if y is not None:
            check_classification_targets(y)
            self._classes = len(np.unique(y))

        return self

    def _set_weights(self, weights):
        """Internal function to set estimator weights.

        Parameters
        ----------
        weights : numpy array of shape (n_estimators,)
            Estimator weights. May be used after the alignment.

        Returns
        -------
        self

        """

        if weights is None:
            self.weights = np.ones([1, self.n_base_estimators_])
        else:
            self.weights = column_or_1d(weights).reshape(1, len(weights))
            if self.weights.shape[1] != self.n_base_estimators_:
                raise RuntimeError("Invalid weights")

            # adjust probability by a factor for integrity （added to 1）
            adjust_factor = self.weights.shape[1] / np.sum(weights)
            self.weights = self.weights * adjust_factor

        return self

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.base_estimators)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.base_estimators[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.base_estimators)

    # noinspection PyMethodParameters
    def _get_param_names(cls):
        # noinspection PyPep8
        """Get parameter names for the estimator

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    # noinspection PyPep8
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        # noinspection PyPep8
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Returns
        -------
        self : object
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class Stacking(BaseAggregator):
    """Meta ensembling, also known as stacking. See
    http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
    for more information

    Parameters
    ----------
    base_estimators: list or numpy array (n_estimators,)
        A list of base classifiers.

    n_folds : int, optional (default=2)
        The number of splits of the training sample.

    keep_original : bool, optional (default=False)
        If True, keep the original features for training and predicting.

    use_proba : bool, optional (default=False)
        If True, use the probability prediction as the new features.

    shuffle_data : bool, optional (default=False)
        If True, shuffle the input data.

    random_state : int, RandomState or None, optional (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    threshold : float in (0, 1), optional (default=None)
        Cut-off value to convert scores into binary labels.

    pre_fitted : bool, optional (default=False)
        Whether the base classifiers are trained. If True, `fit`
        process may be skipped.

    """

    def __init__(
        self,
        base_estimators,
        meta_clf=None,
        n_folds=3,
        keep_original=True,
        use_proba=False,
        shuffle_data=False,
        random_state=None,
        threshold=None,
        pre_fitted=None,
    ):

        super(Stacking, self).__init__(
            base_estimators=base_estimators, pre_fitted=pre_fitted
        )

        # validate input parameters
        if not isinstance(n_folds, int):
            raise ValueError("n_folds must be an integer variable")
        check_parameter(n_folds, low=2, include_left=True, param_name="n_folds")
        self.n_folds = n_folds

        if meta_clf is not None:
            self.meta_clf = copy.deepcopy(meta_clf)
        else:
            self.meta_clf = Pipeline(
                ("imputer", IterativeImputer()), ("output", LogisticRegression())
            )

        # set flags
        self.keep_original = keep_original
        self.use_proba = use_proba
        self.shuffle_data = shuffle_data

        self.random_state = random_state

        if threshold is not None:
            warnings.warn(
                "Stacking does not support threshold setting option. "
                "Please set the threshold in classifiers directly."
            )

        if pre_fitted is not None:
            warnings.warn("Stacking does not support pre_fitted option.")

    def fit(self, X, y):
        """Fit classifier.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """
        self._backup_encoders = {}

        for col in X.columns:
            if X[col].dtype.name not in ["object", "category"]:
                continue

            values = list(X[col].unique())
            values.append("unknown")
            encoder = LabelEncoder().fit(values)
            X.loc[X[col].notna(), col] = encoder.transform(X[col][X[col].notna()])

            self._backup_encoders[col] = encoder

        self.target_encoder = LabelEncoder().fit(y)
        y = self.target_encoder.transform(y)

        # Validate inputs X and y
        X, y = check_X_y(X, y, force_all_finite=False)
        X = check_array(X, force_all_finite=False)
        self._set_n_classes(y)

        n_samples = X.shape[0]

        # initialize matrix for storing newly generated features
        new_features = np.zeros([n_samples, self.n_base_estimators_])

        # build CV datasets
        X_new, y_new, index_lists = split_datasets(
            X,
            y,
            n_folds=self.n_folds,
            shuffle_data=self.shuffle_data,
            random_state=self.random_state,
        )

        # iterate over all base classifiers
        for i, raw_clf in enumerate(self.base_estimators):
            # iterate over all folds
            for j in range(self.n_folds):
                # build train and test index
                full_idx = list(range(n_samples))
                test_idx = index_lists[j]
                train_idx = list_diff(full_idx, test_idx)
                X_train, y_train = X_new[train_idx, :], y_new[train_idx]
                X_test, _ = X_new[test_idx, :], y_new[test_idx]

                # train the classifier
                clf = copy.deepcopy(raw_clf)
                clf.fit(X_train, y_train)

                # generate the new features on the pseudo test set
                if self.use_proba:
                    new_features[test_idx, i] = clf.predict_proba(pd.DataFrame(X_test))[
                        :, 1
                    ]
                else:
                    new_features[test_idx, i] = clf.predict(
                        pd.DataFrame(X_test)
                    ).squeeze()

        # build the new dataset for training
        if self.keep_original:
            X_new_comb = np.concatenate([X_new, new_features], axis=1)
        else:
            X_new_comb = new_features
        y_new_comb = y_new

        # train the meta classifier
        self.meta_clf.fit(X_new_comb, y_new_comb)
        self.fitted_ = True

        # train all base classifiers on the full train dataset
        # iterate over all base classifiers
        for i, clf in enumerate(self.base_estimators):
            clf.fit(X_new, y_new)

        return

    def _process_data(self, X):
        """Internal class for both `predict` and `predict_proba`

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_new_comb : Numpy array
            The processed dataset of X.
        """
        check_is_fitted(self, ["fitted_"])
        for col in self._backup_encoders:
            eval_data = X[col][X[col].notna()]
            inf_values = [
                x if x in self._backup_encoders[col].classes_ else "unknown"
                for x in eval_data
            ]

            X.loc[X[col].notna(), col] = self._backup_encoders[col].transform(
                inf_values
            )

        X = check_array(X, force_all_finite=False)
        n_samples = X.shape[0]

        # initialize matrix for storing newly generated features
        new_features = np.zeros([n_samples, self.n_base_estimators_])

        # build the new features for unknown samples
        # iterate over all base classifiers
        for i, clf in enumerate(self.base_estimators):
            # generate the new features on the test set
            if self.use_proba:
                new_features[:, i] = clf.predict_proba(X)[:, 1]
            else:
                new_features[:, i] = clf.predict(X).squeeze()

        # build the new dataset for unknown samples
        if self.keep_original:
            X_new_comb = np.concatenate([X, new_features], axis=1)
        else:
            X_new_comb = new_features

        return X_new_comb

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        X_new_comb = self._process_data(X)
        return self.meta_clf.predict(X_new_comb)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : numpy array of shape (n_samples,)
            The class probabilities of the input samples.
            Classes are ordered by lexicographic order.
        """
        X_new_comb = self._process_data(X)
        return self.meta_clf.predict_proba(X_new_comb)

    def fit_predict(self, X, y):
        """Fit estimator and predict on X

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        raise NotImplementedError(
            "fit_predict should not be used in supervised learning models."
        )


class SimpleClassifierAggregator(BaseAggregator):
    """A collection of simple classifier combination methods.

    Parameters
    ----------
    base_estimators: list or numpy array (n_estimators,)
        A list of base classifiers.

    method : str, optional (default='average')
        Combination method: {'average', 'maximization', 'majority vote',
        'median'}. Pass in weights of classifier for weighted version.

    threshold : float in (0, 1), optional (default=0.5)
        Cut-off value to convert scores into binary labels.

    weights : numpy array of shape (1, n_classifiers)
        Classifier weights.

    pre_fitted : bool, optional (default=False)
        Whether the base classifiers are trained. If True, `fit`
        process may be skipped.
    """

    def __init__(
        self,
        base_estimators,
        method="average",
        threshold=0.5,
        weights=None,
        pre_fitted=False,
    ):

        super(SimpleClassifierAggregator, self).__init__(
            base_estimators=base_estimators, pre_fitted=pre_fitted
        )

        # validate input parameters
        if method not in ["average", "maximization", "majority_vote", "median"]:
            raise ValueError(f"{method} is not a valid parameter.")

        self.method = method
        check_parameter(
            threshold,
            0,
            1,
            include_left=False,
            include_right=False,
            param_name="threshold",
        )
        self.threshold = threshold

        # set estimator weights
        self._set_weights(weights)

    def fit(self, X, y):
        """Fit classifier.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """
        self._backup_encoders = {}

        for col in X.columns:
            if X[col].dtype.name not in ["object", "category"]:
                continue

            values = list(X[col].unique())
            values.append("unknown")
            encoder = LabelEncoder().fit(values)
            X.loc[X[col].notna(), col] = encoder.transform(X[col][X[col].notna()])

            self._backup_encoders[col] = encoder

        self.target_encoder = LabelEncoder().fit(y)
        y = self.target_encoder.transform(y)

        # Validate inputs X and y
        X, y = check_X_y(X, y, force_all_finite=False)
        X = check_array(X, force_all_finite=False)
        self._set_n_classes(y)

        if self.pre_fitted:
            return
        else:
            for clf in self.base_estimators:
                clf.fit(X, y)
                clf.fitted_ = True
            return

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        for col in self._backup_encoders:
            eval_data = X[col][X[col].notna()]
            inf_values = [
                x if x in self._backup_encoders[col].classes_ else "unknown"
                for x in eval_data
            ]

            X.loc[X[col].notna(), col] = self._backup_encoders[col].transform(
                inf_values
            )

        X = check_array(X, force_all_finite=False)

        all_scores = np.zeros([X.shape[0], self.n_base_estimators_])

        for i, clf in enumerate(self.base_estimators):
            if clf.fitted_ is not True and self.pre_fitted is False:
                ValueError("Classifier should be fitted first!")
            else:
                if hasattr(clf, "predict"):
                    all_scores[:, i] = clf.predict(X)
                else:
                    raise ValueError(f"{clf} does not have predict.")

        if self.method == "average":
            agg_score = average(all_scores, estimator_weights=self.weights)
        if self.method == "maximization":
            agg_score = maximization(all_scores)
        if self.method == "majority_vote":
            agg_score = majority_vote(all_scores, weights=self.weights)
        if self.method == "median":
            agg_score = median(all_scores)

        return (agg_score >= self.threshold).astype("int").ravel()

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : numpy array of shape (n_samples,)
            The class probabilities of the input samples.
            Classes are ordered by lexicographic order.
        """
        for col in self._backup_encoders:
            eval_data = X[col][X[col].notna()]
            inf_values = [
                x if x in self._backup_encoders[col].classes_ else "unknown"
                for x in eval_data
            ]

            X.loc[X[col].notna(), col] = self._backup_encoders[col].transform(
                inf_values
            )

        X = check_array(X, force_all_finite=False)
        all_scores = np.zeros([X.shape[0], self._classes, self.n_base_estimators_])

        for i in range(self.n_base_estimators_):
            clf = self.base_estimators[i]
            if clf.fitted_ is not True and self.pre_fitted is False:
                ValueError("Classifier should be fitted first!")
            else:
                if hasattr(clf, "predict_proba"):
                    all_scores[:, :, i] = clf.predict_proba(X)
                else:
                    raise ValueError(f"{clf} does not have predict_proba.")

        if self.method == "average":
            return np.mean(all_scores * self.weights, axis=2)
        if self.method == "maximization":
            scores = np.max(all_scores * self.weights, axis=2)
            return score_to_proba(scores)
        if self.method == "majority_vote":
            Warning(
                "average method is invoked for predict_proba as"
                "probability is not continuous"
            )
            return np.mean(all_scores * self.weights, axis=2)
        if self.method == "median":
            Warning(
                "average method is invoked for predict_proba as"
                "probability is not continuous"
            )
            return np.mean(all_scores * self.weights, axis=2)

    def fit_predict(self, X, y):
        """Fit estimator and predict on X

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).

        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Class labels for each data sample.
        """
        raise NotImplementedError(
            "fit_predict should not be used in supervised learning models."
        )


def split_datasets(X, y, n_folds=3, shuffle_data=False, random_state=None):
    """Utility function to split the data for stacking. The data is split
    into n_folds with roughly equal rough size.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input samples.

    y : numpy array of shape (n_samples,)
        The ground truth of the input samples (labels).

    n_folds : int, optional (default=3)
        The number of splits of the training sample.

    shuffle_data : bool, optional (default=False)
        If True, shuffle the input data.

    random_state : RandomState, optional (default=None)
        A random number generator instance to define the state of the random
        permutations generator.

    Returns
    -------
    X : numpy array of shape (n_samples, n_features)
        The input samples. If shuffle_data, return the shuffled data.

    y : numpy array of shape (n_samples,)
        The ground truth of the input samples (labels). If shuffle_data,
        return the shuffled data.

    index_lists : list of list
        The list of indexes of each fold regarding the returned X and y.
        For instance, index_lists[0] contains the indexes of fold 0.

    """

    if not isinstance(n_folds, int):
        raise ValueError("n_folds must be an integer variable")
    check_parameter(n_folds, low=2, include_left=True, param_name="n_folds")

    random_state = check_random_state(random_state)

    if shuffle_data:
        X, y = shuffle(X, y, random_state=random_state)

    idx_length = len(y)
    idx_list = list(range(idx_length))

    avg_length = int(idx_length / n_folds)

    index_lists = []
    for i in range(n_folds - 1):
        index_lists.append(idx_list[i * avg_length : (i + 1) * avg_length])

    index_lists.append(idx_list[(n_folds - 1) * avg_length :])

    return X, y, index_lists


def list_diff(first_list, second_list):
    """Utility function to calculate list difference (first_list-second_list)
    Parameters
    ----------
    first_list : list
        First list.
    second_list : list
        Second list.
    Returns
    -------
    diff : different elements.
    """
    second_list = set(second_list)
    return [item for item in first_list if item not in second_list]


def average(scores, estimator_weights=None):
    """Combination method to merge the scores from multiple estimators
    by taking the average.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    estimator_weights : numpy array of shape (1, n_estimators)
        If specified, using weighted average.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined scores.

    """
    scores = check_array(scores)

    if estimator_weights is not None:
        if estimator_weights.shape != (1, scores.shape[1]):
            raise ValueError(
                "Bad input shape of estimator_weight: (1, {score_shape}),"
                "and {estimator_weights} received".format(
                    score_shape=scores.shape[1],
                    estimator_weights=estimator_weights.shape,
                )
            )

        # (d1*w1 + d2*w2 + ...+ dn*wn)/(w1+w2+...+wn)
        # generated weighted scores
        scores = np.sum(np.multiply(scores, estimator_weights), axis=1) / np.sum(
            estimator_weights
        )
        return scores.ravel()

    else:
        return np.mean(scores, axis=1).ravel()


def maximization(scores):
    """Combination method to merge the scores from multiple estimators
    by taking the maximum.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined scores.

    """

    scores = check_array(scores)
    return np.max(scores, axis=1).ravel()


def median(scores):
    """Combination method to merge the scores from multiple estimators
    by taking the median.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined scores.

    """

    scores = check_array(scores)
    return np.median(scores, axis=1).ravel()


def majority_vote(scores, n_classes=2, weights=None):
    """Combination method to merge the scores from multiple estimators
    by majority vote.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    n_classes : int, optional (default=2)
        The number of classes in scores matrix

    weights : numpy array of shape (1, n_estimators)
        If specified, using weighted majority weight.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined scores.

    """

    scores = check_array(scores)

    # assert only discrete scores are combined with majority vote
    check_classification_targets(scores)

    n_samples, n_estimators = scores.shape[0], scores.shape[1]

    vote_results = np.zeros(
        [
            n_samples,
        ]
    )

    if weights is not None:
        if scores.shape[1] != weights.shape[1]:
            raise ValueError("invalid weights")

    # equal weights if not set
    else:
        weights = np.ones([1, n_estimators])

    for i in range(n_samples):
        vote_results[i] = weighted_mode(scores[i, :], weights)[0][0]

    return vote_results.ravel()


def score_to_proba(scores):
    """Internal function to random score matrix into probability.
    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_classes)
        Raw score matrix.
    Returns
    -------
    proba : numpy array of shape (n_samples, n_classes)
        Scaled probability matrix.
    """

    scores_sum = np.sum(scores, axis=1).reshape(scores.shape[0], 1)
    scores_sum_broadcast = np.broadcast_to(
        scores_sum, (scores.shape[0], scores.shape[1])
    )
    proba = scores / scores_sum_broadcast
    return proba
