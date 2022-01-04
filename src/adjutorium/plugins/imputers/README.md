# Imputation Plugins

### :information_source: About


Dataset imputation is the process of replacing missing data with substituted values.


This module provides the default imputation plugins supported by Adjutorium.The library automatically loads every file that follows the pattern `*_plugin.py` and exports a class derived from the [`ImputerPlugin`](base.py) interface.

### :zap: Plugins
The following table contains the default imputation plugins:

| Strategy | Description| Based on| Code | Tests|
|--- | --- | --- | --- | --- |
|**HyperImpute**|Generalized Iterative Imputer|| [`plugin_hyperimpute.py`](plugin_hyperimpute.py) | [`test_hyperimpute.py`](../../../../tests/plugins/imputers/test_hyperimpute.py) |
|**Mean**|Replace the missing values using the mean along each column|[`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)| [`plugin_mean.py`](plugin_mean.py) | [`test_mean.py`](../../../../tests/plugins/imputers/test_mean.py) |
|**Median**|Replace the missing values using the median along each column|[`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)| [`plugin_median.py`](plugin_median.py) | [`test_median.py`](../../../../tests/plugins/imputers/test_median.py)|
|**Most-frequent**|Replace the missing values using the most frequent value along each column|[`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)|[`plugin_most_freq.py`](plugin_most_freq.py) | [`test_most_freq.py`](../../../../tests/plugins/imputers/test_most_freq.py) |
|**MissForest**|Iterative imputation method based on Random Forests| [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)| [`plugin_missforest.py`](plugin_missforest.py) |[`test_missforest.py`](../../../../tests/plugins/imputers/test_missforest.py) |
|**ICE**| Iterative imputation method based on regularized linear regression | [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)| [`plugin_ice.py`](plugin_ice.py)| [`test_ice.py`](../../../../tests/plugins/imputers/test_ice.py)|
|**MICE**| Multiple imputations based on ICE| [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)| [`plugin_mice.py`](plugin_mice.py) |[`test_mice.py`](../../../../tests/plugins/imputers/test_mice.py) |
|**SoftImpute**|Low-rank matrix approximation via nuclear-norm regularization| [`Original paper`](https://jmlr.org/papers/volume16/hastie15a/hastie15a.pdf)| [`plugin_softimpute.py`](plugin_softimpute.py)|[`test_softimpute.py`](../../../../tests/plugins/imputers/test_softimpute.py) |
|**EM**|Iterative procedure which uses other variables to impute a value (Expectation), then checks whether that is the value most likely (Maximization)|[`EM imputation algorithm`](https://joon3216.github.io/research_materials/2019/em_imputation.html)|[`plugin_em.py`](plugin_em.py) |[`test_em.py`](../../../../tests/plugins/imputers/test_em.py) |
|**Sinkhorn**|Based on the Optimal transport distances between random batches|[`Original paper`](https://arxiv.org/pdf/2002.03860.pdf)|[`plugin_sinkhorn.py`](plugin_sinkhorn.py) | [`test_sinkhorn.py`](../../../../tests/plugins/imputers/test_sinkhorn.py)|
