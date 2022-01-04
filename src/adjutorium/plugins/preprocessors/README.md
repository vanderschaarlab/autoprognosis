# Preprocessing Plugins

### :information_source: About

Preprocessing datasets is a common requirement for many machine learning estimators. The techniques include:
 - dimensionality reduction: the process of reducing the dimension of your feature set.
 - feature scaling: the process of normalizing the range or the shape of the features in the dataset.

This module provides the default preprocessing plugins supported by Adjutorium.The library automatically loads every file that follows the pattern `*_plugin.py` and exports a class derived from the [`PreprocessorPlugin`](base.py) interface.

### :zap: Plugins
The following table contains the default preprocessing plugins:

| Strategy | Purpose | Based on| Code | Tests|
|--- | --- | --- | --- | --- |
|**Fast ICA**| dimensionality reduction | [sklearn.decomposition.FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html) |[`plugin_fast_ica.py`](dimensionality_reduction/plugin_fast_ica.py) | [`test_fast_ica.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_fast_ica.py) |
|**Feature agglomeration**| dimensionality reduction | [sklearn.cluster.FeatureAgglomeration](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html) |[`plugin_feature_agglomeration.py`](dimensionality_reduction/plugin_feature_agglomeration.py) | [`test_feature_agglomeration.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_feature_agglomeration.py) |
|**Feature normalization**| feature scaling | [sklearn.preprocessing.Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) |[`plugin_feature_normalizer.py`](feature_scaling/plugin_feature_normalizer.py) | [`test_feature_normalizer.py`](../../../../tests/plugins/preprocessors/feature_scaling/test_feature_normalizer.py) |
|**Feature selection**| dimensionality reduction  | [sklearn.feature_selection.SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) |[`plugin_feature_selection.py`](dimensionality_reduction/plugin_feature_selection.py) | [`test_feature_selection.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_feature_selection.py) |
|**Gaussian Random Projection**| dimensionality reduction  | [sklearn.random_projection.GaussianRandomProjection](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html) |[`plugin_gauss_projection.py`](dimensionality_reduction/plugin_gauss_projection.py) | [`test_gauss_projection.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_gauss_projection.py) |
|**Kernel PCA**| dimensionality reduction  | [sklearn.decomposition.KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) |[`plugin_kernel_pca.py`](dimensionality_reduction/plugin_kernel_pca.py) | [`test_kernel_pca.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_kernel_pca.py) |
|**Linear SVM**| dimensionality reduction  | [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) |[`plugin_linear_svm.py`](dimensionality_reduction/plugin_linear_svm.py) | [`test_linear_svm.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_linear_svm.py) |
|**Max absolute scaler**| feature scaling  | [sklearn.preprocessing.MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) |[`plugin_maxabs_scaler.py`](feature_scaling/plugin_maxabs_scaler.py) | [`test_maxabs_scaler.py`](../../../../tests/plugins/preprocessors/feature_scaling/test_maxabs_scaler.py) |
|**MinMax scaler**| feature scaling  | [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) |[`plugin_minmax_scaler.py`](feature_scaling/plugin_minmax_scaler.py) | [`test_minmax_scaler.py`](../../../../tests/plugins/preprocessors/feature_scaling/test_minmax_scaler.py) |
|**Normal transform**| feature scaling  | [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html) |[`plugin_normal_transform.py`](feature_scaling/plugin_normal_transform.py) | [`test_normal_transform.py`](../../../../tests/plugins/preprocessors/feature_scaling/test_normal_transform.py) |
|**Nystroem**| dimensionality reduction  | [sklearn.kernel_approximation.Nystroem](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html) |[`plugin_nystroem.py`](dimensionality_reduction/plugin_nystroem.py) | [`test_nystroem.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_nystroem.py) |
|**PCA**| dimensionality reduction  | [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) |[`plugin_pca.py`](dimensionality_reduction/plugin_pca.py) | [`test_pca.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_pca.py) |
|**Random kitchen sinks**| dimensionality reduction  | [sklearn.kernel_approximation.RBFSampler](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html) |[`plugin_random_kitchen_sinks.py`](dimensionality_reduction/plugin_random_kitchen_sinks.py) | [`test_random_kitchen_sinks.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_random_kitchen_sink.py) |
|**Standard scaler**| feature scaling  | [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) |[`plugin_scaler.py`](feature_scaling/plugin_scaler.py) | [`test_scaler.py`](../../../../tests/plugins/preprocessors/feature_scaling/test_scaler.py) |
|**Select FDR**| dimensionality reduction | [sklearn.feature_selection.SelectFdr](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html) |[`plugin_select_fdr.py`](dimensionality_reduction/plugin_select_fdr.py) | [`test_select_fdr.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_select_fdr.py) |
|**Uniform transform**| feature scaling  | [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html) |[`plugin_uniform_transform.py`](feature_scaling/plugin_uniform_transform.py) | [`test_uniform_transform.py`](../../../../tests/plugins/preprocessors/feature_scaling/test_uniform_transform.py) |
|**Variance threshold**| dimensionality reduction  | [sklearn.feature_selection.VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html) |[`plugin_variance_threshold.py`](dimensionality_reduction/plugin_variance_threshold.py) | [`test_variance_threshold.py`](../../../../tests/plugins/preprocessors/dimensionality_reduction/test_variance_threshold.py) |

### :hammer: Writing a new preprocessing plugin
Every **Adjutorium plugin** must implement the **`Plugin`** interface provided by [`adjutorium/plugins/core/base_plugin.py`](../core/base_plugin.py).

Each **Adjutorium preprocessing plugin** must implement the **`PreprocessorPlugin`** interface provided by [`adjutorium/plugins/preprocessors/base.py`](base.py)

:heavy_exclamation_mark: __Warning__ : If a plugin doesn't override all the abstract methods, it won't be loaded by the library.



Every preprocessing plugin **must implement** the following methods:
- *name()* - a static method that returns the name of the plugin. e.g., variance_threshold, minmax_scaler etc.

- *hyperparameter_space()* - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `params.Params` derived objects.

- *_fit()* - internal implementation, called by the `fit` method.
- *_transform()* - internal implementation, called by the `transform` method.

### :cyclone: Example: Adding a new plugin

```
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2

custom_select_fpr = "custom_select_fpr"

class NewPlugin(PreprocessorPlugin):
    def __init__(self):
        super().__init__()
        self._model = SelectFpr(chi2)

    @staticmethod
    def name():
        return custom_select_fpr

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []

    def _fit(self, *args, **kwargs):
        self._model.fit(*args, **kwargs)

        return self

    def _transform(self, *args, **kwargs):
        return self._model.transform(*args, **kwargs)

preprocessors.add(custom_select_fpr, NewPlugin)

assert preprocessors.get(custom_select_fpr) is not None
```
