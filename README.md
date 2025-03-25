# AutoPrognosis - A system for automating the design of predictive modeling pipelines tailored for clinical prognosis.


<div align="center">


[![Test In Colab](https://img.shields.io/badge/Tutorial-Model%20Search-orange)](https://colab.research.google.com/drive/1sFVnnxjRMCNVIn-Ikc--Ja44U0Ll4joY?usp=sharing)
[![Test In Colab](https://img.shields.io/badge/Tutorial-Build%20a%20Demonstrator-orange)](https://colab.research.google.com/drive/1ZwjD9RkosCtboyblH4C8sQV1DuGY1H2X?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2210.12090-b31b1b.svg)](https://arxiv.org/abs/2210.12090)


[![Tests](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_pr.yml/badge.svg)](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_pr.yml)
[![Tests](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_full.yml/badge.svg)](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_full.yml)
<!-- [![Tests R](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_R.yml/badge.svg)](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_R.yml) -->
[![Tutorials](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_tutorials.yml/badge.svg)](https://github.com/vanderschaarlab/autoprognosis/actions/workflows/test_tutorials.yml)
[![Documentation Status](https://readthedocs.org/projects/autoprognosis/badge/?version=latest)](https://autoprognosis.readthedocs.io/en/latest/?badge=latest)

[![](https://pepy.tech/badge/autoprognosis)](https://pypi.org/project/autoprognosis/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vanderschaarlab/autoprognosis/blob/main/LICENSE)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/vanderschaarlab/shared_invite/zt-1pzy8z7ti-zVsUPHAKTgCd1UoY8XtTEw)

</div>


![image](https://github.com/vanderschaarlab/autoprognosis/raw/main/docs/arch.png "AutoPrognosis")

## :key: Features

- :rocket: Automatically learns ensembles of pipelines for classification, regression or survival analysis tasks.
- :cyclone: Easy to extend pluginable architecture.
- :fire: Interpretability and uncertainty quantification tools.
- :adhesive_bandage: Data imputation using [HyperImpute](https://github.com/vanderschaarlab/hyperimpute).
- :zap: Build demonstrators using [Streamlit](https://streamlit.io/).
- :notebook: [Python](#high_brightness-tutorials) and [R tutorials](https://github.com/vanderschaarlab/autoprognosis/tree/main/tutorials/bindings/R) available.
- :book: [Read the docs](https://autoprognosis.readthedocs.io/)

__Note__ : The R bindings have been tested using R version 4.2 and Python 3.8.

## :rocket: Installation

#### Using pip

The library can be installed from PyPI using
```bash
$ pip install autoprognosis
```
or from source, using
```bash
$ pip install .
```
### Redis (Optional, but recommended)
AutoPrognosis can use Redis as a backend to improve the performance and quality of the searches.

For that, install the redis-server package following the steps described on the [official site](https://redis.io/topics/quickstart).

## Environment variables
The library can be configured from a set of environment variables.

| Variable       | Description                                                     |
|----------------|-----------------------------------------------------------------|
| `N_OPT_JOBS`     | Number of cores to use for hyperparameter search. Default : 1 |
| `N_LEARNER_JOBS` | Number of cores to use by inidividual learners. Default: all cpus      |
| `REDIS_HOST`     | IP address for the Redis database. Default 127.0.0.1            |
| `REDIS_PORT`     | Redis port. Default: 6379                                       |

_Example_: `export N_OPT_JOBS = 2` to use 2 cores for hyperparam search.

## :boom: Sample Usage

__Advanced Python tutorials__ can be found in the [Python tutorials section](#high_brightness-tutorials).


__R examples__ can be found in the [R tutorials section](https://github.com/vanderschaarlab/autoprognosis/tree/main/tutorials/bindings/R).

List the available classifiers
```python
from autoprognosis.plugins.prediction.classifiers import Classifiers
print(Classifiers().list_available())
```

Create a study for classifiers
```python
from sklearn.datasets import load_breast_cancer

from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_estimator


X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

df = X.copy()
df["target"] = Y

study_name = "example"

study = ClassifierStudy(
    study_name=study_name,
    dataset=df,  # pandas DataFrame
    target="target",  # the label column in the dataset
)
model = study.fit()

# Predict the probabilities of each class using the model
model.predict_proba(X)
```


__(Advanced)__ Customize the study for classifiers
```python
from pathlib import Path

from sklearn.datasets import load_breast_cancer

from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_estimator


X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

df = X.copy()
df["target"] = Y

workspace = Path("workspace")
study_name = "example"

study = ClassifierStudy(
    study_name=study_name,
    dataset=df,  # pandas DataFrame
    target="target",  # the label column in the dataset
    num_iter=100,  # how many trials to do for each candidate
    timeout=60,  # seconds
    classifiers=["logistic_regression", "lda", "qda"],
    workspace=workspace,
)

study.run()

output = workspace / study_name / "model.p"
model = load_model_from_file(output)

# <model> contains the optimal architecture, but the model is not trained yet. You need to call fit() to use it.
# This way, we can further benchmark the selected model on the training set.
metrics = evaluate_estimator(model, X, Y)

print(f"model {model.name()} -> {metrics['str']}")

# Train the model
model.fit(X, Y)

# Predict the probabilities of each class using the model
model.predict_proba(X)
```

List the available regressors
```python
from autoprognosis.plugins.prediction.regression import Regression
print(Regression().list_available())
```

Create a Regression study
```python
# third party
import pandas as pd

# autoprognosis absolute
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_regression
from autoprognosis.studies.regression import RegressionStudy

# Load dataset
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    header=None,
    sep="\\t",
)
last_col = df.columns[-1]
y = df[last_col]
X = df.drop(columns=[last_col])

df = X.copy()
df["target"] = y

# Search the model
study_name="regression_example"
study = RegressionStudy(
    study_name=study_name,
    dataset=df,  # pandas DataFrame
    target="target",  # the label column in the dataset
)
model = study.fit()

# Predict using the model
model.predict(X)
```

__Advanced__ Customize the Regression study
```python
# stdlib
from pathlib import Path

# third party
import pandas as pd

# autoprognosis absolute
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_regression
from autoprognosis.studies.regression import RegressionStudy

# Load dataset
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    header=None,
    sep="\\t",
)
last_col = df.columns[-1]
y = df[last_col]
X = df.drop(columns=[last_col])

df = X.copy()
df["target"] = y

# Search the model
workspace = Path("workspace")
workspace.mkdir(parents=True, exist_ok=True)

study_name="regression_example"
study = RegressionStudy(
    study_name=study_name,
    dataset=df,  # pandas DataFrame
    target="target",  # the label column in the dataset
    num_iter=10,  # how many trials to do for each candidate. Default: 50
    num_study_iter=2,  # how many outer iterations to do. Default: 5
    timeout=50,  # timeout for optimization for each classfier. Default: 600 seconds
    regressors=["linear_regression", "xgboost_regressor"],
    workspace=workspace,
)

study.run()

# Test the model
output = workspace / study_name / "model.p"

model = load_model_from_file(output)
# <model> contains the optimal architecture, but the model is not trained yet. You need to call fit() to use it.
# This way, we can further benchmark the selected model on the training set.

metrics = evaluate_regression(model, X, y)

print(f"Model {model.name()} score: {metrics['str']}")

# Train the model
model.fit(X, y)


# Predict using the model
model.predict(X)
```

List available survival analysis estimators
```python
from autoprognosis.plugins.prediction.risk_estimation import RiskEstimation
print(RiskEstimation().list_available())
```
Create a Survival analysis study
```python
# third party
import numpy as np
from pycox import datasets

# autoprognosis absolute
from autoprognosis.studies.risk_estimation import RiskEstimationStudy
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_survival_estimator

df = datasets.gbsg.read_df()
df = df[df["duration"] > 0]

X = df.drop(columns = ["duration"])
T = df["duration"]
Y = df["event"]

eval_time_horizons = np.linspace(T.min(), T.max(), 5)[1:-1]

study_name = "example_risks"

study = RiskEstimationStudy(
    study_name=study_name,
    dataset=df,
    target="event",
    time_to_event="duration",
    time_horizons=eval_time_horizons,
)

model = study.fit()

# Predict using the model
model.predict(X, eval_time_horizons)
```

__Advanced__ Customize the Survival analysis study
```python
# stdlib
import os
from pathlib import Path

# third party
import numpy as np
from pycox import datasets

# autoprognosis absolute
from autoprognosis.studies.risk_estimation import RiskEstimationStudy
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_survival_estimator

df = datasets.gbsg.read_df()
df = df[df["duration"] > 0]

X = df.drop(columns = ["duration"])
T = df["duration"]
Y = df["event"]

eval_time_horizons = np.linspace(T.min(), T.max(), 5)[1:-1]

workspace = Path("workspace")
study_name = "example_risks"

study = RiskEstimationStudy(
    study_name=study_name,
    dataset=df,
    target="event",
    time_to_event="duration",
    time_horizons=eval_time_horizons,
    num_iter=10,
    num_study_iter=1,
    timeout=10,
    risk_estimators=["cox_ph", "survival_xgboost"],
    score_threshold=0.5,
    workspace=workspace,
)

study.run()

output = workspace / study_name / "model.p"

model = load_model_from_file(output)
# <model> contains the optimal architecture, but the model is not trained yet. You need to call fit() to use it.
# This way, we can further benchmark the selected model on the training set.

metrics = evaluate_survival_estimator(model, X, T, Y, eval_time_horizons)

print(f"Model {model.name()} score: {metrics['str']}")

# Train the model
model.fit(X, T, Y)

# Predict using the model
model.predict(X, eval_time_horizons)
```

## :high_brightness: Tutorials

### Plugins


- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QO7K3JqW8l4pgVSLxjVezTu5IfD9yHB-?usp=sharing) [ Imputation](tutorials/plugins/tutorial_00_imputation_plugins.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WQGZXQkQs0Wg5stB9fk-RvYey35ADIZu?usp=sharing)[ Preprocessing](tutorials/plugins/tutorial_01_preprocessing_plugins.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WTzO_2hqaEOvvATHPSIcW220xc1WaJlC?usp=sharing)[ Classification](tutorials/plugins/tutorial_02_classification_plugins.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17bLtKUjN8ilHw4Cm7-53kiC0vCJO_pVb?usp=sharing)[ Pipelines](tutorials/plugins/tutorial_03_pipelines.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K0yVwm4jQrXRbMKJ-em7tTYgHXWtoK5c?usp=sharing)[ Interpretability](tutorials/plugins/tutorial_04_interpretability.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bY4CbiqMe2uoqeUu2d49aIdYRbtP156X?usp=sharing)[ Survival Analysis](tutorials/plugins/tutorial_05_survival_analysis_plugins.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UK6WbsviT5nOQ_BAHSFYIjhpKtwppnUU?usp=sharing)[ Regression](tutorials/plugins/tutorial_06_regression_plugins.ipynb)

### AutoML


 - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-lPuQAtjHESl32ahFQYsFl8ujAnDWxEJ?usp=sharing)[ Classification tasks](tutorials/automl/tutorial_00_classification_study.ipynb)
 - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16UDaA3F5JGw_YVY8XlYqWjfxcUV1OHJo?usp=sharing)[ Classification tasks with imputation](tutorials/automl/tutorial_01_automl_classification_with_imputation.ipynb)
 - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DtZCqebhaYdKB3ci5dr3hT0KvZPaTUOi?usp=sharing)[ Survival analysis tasks](tutorials/automl/tutorial_02_survival_analysis_study.ipynb)
 - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sFVnnxjRMCNVIn-Ikc--Ja44U0Ll4joY?usp=sharing)[ Survival analysis tasks with imputation](tutorials/automl/tutorial_03_automl_survival_analysis_with_imputation.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HLhWI-tRZn4e9ijQ6iEIuppuDszgWkCC?usp=sharing)[ Regression tasks](tutorials/automl/tutorial_04_regression.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eHw1l79_m3vq9y-0WpllCMBSD7DQajWO?usp=sharing)[ Classifiers with explainers](tutorials/automl/tutorial_05_classification_with_explainers.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dm3cRmo-jD6x7V5WePciDpcauqtUt6lS?usp=sharing)[ Multiple imputation example](tutorials/automl/tutorial_06_automl_multiple_imputation_example.ipynb)

### Building a demonstrator


 - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lqbElEVJa2Q0JDsXPgb8K_QUTDcZvUQq?usp=sharing)[ Classification demonstrator](tutorials/demonstrators/tutorial_00_build_a_demonstrator_classification.ipynb)
 - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZwjD9RkosCtboyblH4C8sQV1DuGY1H2X?usp=sharing)[ Survival analysis demonstrator](tutorials/demonstrators/tutorial_01_build_a_demonstrator_survival_analysis.ipynb)

### AutoPrognosis 101 Tutorial Series
 - [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1axBQRnGCeh6vqhisfMNBLe9jHNKhVvG2)[ 00. Run a classification study](https://colab.research.google.com/drive/1axBQRnGCeh6vqhisfMNBLe9jHNKhVvG2)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RIxuYuFyTlaE1RZ0-y-W2kx54_vlWYTL)[ 01. Run a regression study](https://colab.research.google.com/drive/1RIxuYuFyTlaE1RZ0-y-W2kx54_vlWYTL)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13shRlJgADizDgWJP0tN8HhIVfT8vnCeS)[ 02. Run a survival analysis study](https://colab.research.google.com/drive/13shRlJgADizDgWJP0tN8HhIVfT8vnCeS)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/137KvWXXYZXaajbrVBW_vEeCEJmyXA4OZ)[ 03. Run a study and interpret the model](https://colab.research.google.com/drive/137KvWXXYZXaajbrVBW_vEeCEJmyXA4OZ)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13QMhuDfuXJNllvCa09T27YSyaJlmpBjP)[ 04. What’s a plugin? Survival analysis example](https://colab.research.google.com/drive/13QMhuDfuXJNllvCa09T27YSyaJlmpBjP)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nl0ZUWbglmJQzTowLoSOz2pmHIzBZm-z)[ 05. Pipelines](https://colab.research.google.com/drive/1nl0ZUWbglmJQzTowLoSOz2pmHIzBZm-z)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Y47ortMzmqvc0JPvptIsiNiHFoynni-j)[ 06. [Advanced] Creating your own plugin: preprocessing example](https://colab.research.google.com/drive/1Y47ortMzmqvc0JPvptIsiNiHFoynni-j)


## :zap: Plugins

### Imputation methods


```python
from autoprognosis.plugins.imputers import  Imputers

imputer = Imputers().get(<NAME>)
```

| Name | Description |
|--- | --- |
|**hyperimpute**|Iterative imputer using both regression and classification methods based on linear models, trees, XGBoost, CatBoost and neural nets|
|**mean**|Replace the missing values using the mean along each column with [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)|
|**median**|Replace the missing values using the median along each column with [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) |
|**most_frequent**|Replace the missing values using the most frequent value along each column with [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)|
|**missforest**|Iterative imputation method based on Random Forests using [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)|
|**ice**| Iterative imputation method based on regularized linear regression using [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)|
|**mice**| Multiple imputations based on ICE using [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)|
|**softimpute**|  [`Low-rank matrix approximation via nuclear-norm regularization`](https://jmlr.org/papers/volume16/hastie15a/hastie15a.pdf)| [`plugin_softimpute.py`](src/hyperimpute/plugins/imputers/plugin_softimpute.py)|
|**EM**|Iterative procedure which uses other variables to impute a value (Expectation), then checks whether that is the value most likely (Maximization) - [`EM imputation algorithm`](https://joon3216.github.io/research_materials/2019/em_imputation.html)|
|**gain**|[`GAIN: Missing Data Imputation using Generative Adversarial Nets`](https://arxiv.org/abs/1806.02920)|


### Preprocessing methods
```python
from autoprognosis.plugins.preprocessors import Preprocessors

preprocessor = Preprocessors().get(<NAME>)
```
| Name | Description |
|--- | --- |
| **maxabs_scaler**  | Scale each feature by its maximum absolute value. [`MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)|
| **scaler** |Standardize features by removing the mean and scaling to unit variance. - [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)|
|**feature_normalizer** | Normalize samples individually to unit norm. [`Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer)|
|**normal_transform** |Transform features using quantiles information.[`QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer)|
|**uniform_transform** |Transform features using quantiles information.[`QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer)|
|**minmax_scaler** |Transform features by scaling each feature to a given range.[`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)|


### Classification
```python
from autoprognosis.plugins.prediction.classifiers import Classifiers

classifier = Classifiers().get(<NAME>)
```

| Name | Description |
|--- | --- |
| **neural_nets**  | PyTorch based neural net classifier.|
| **logistic_regression**  | [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)|
| **catboost**  |Gradient boosting on decision trees - [`CatBoost`](https://catboost.ai/)|
| **random_forest**  | A random forest classifier. [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)|
| **tabnet**  |[`TabNet : Attentive Interpretable Tabular Learning`](https://github.com/dreamquark-ai/tabnet)|
| **xgboost**  |[`XGBoostClassifier`](https://xgboost.readthedocs.io/en/stable/)|


### Survival Analysis
```python
from autoprognosis.plugins.prediction.risk_estimation import RiskEstimation

predictor = RiskEstimation().get(<NAME>)
```

| Name | Description |
|--- | --- |
| **survival_xgboost**  | [`XGBoost Survival Embeddings`](https://github.com/loft-br/xgboost-survival-embeddings)|
| **loglogistic_aft**  | [` Log-Logistic AFT model`](https://lifelines.readthedocs.io/en/latest/fitters/regression/LogLogisticAFTFitter.html)|
| **deephit**  | [`DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks`](https://github.com/chl8856/DeepHit)|
| **cox_ph**  | [`Cox’s proportional hazard model`](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html)|
| **weibull_aft**  | [`Weibull AFT model.`](https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html)|
| **lognormal_aft**  | [`Log-Normal AFT model`](https://lifelines.readthedocs.io/en/latest/fitters/regression/LogNormalAFTFitter.html)|
| **coxnet**  | [`CoxNet is a Cox proportional hazards model also referred to as DeepSurv`](https://github.com/havakv/pycox)|

### Regression
```python
from autoprognosis.plugins.prediction.regression import Regression

regressor = Regression().get(<NAME>)
```

| Name | Description |
|--- | --- |
| **tabnet_regressor**  |[`TabNet : Attentive Interpretable Tabular Learning`](https://github.com/dreamquark-ai/tabnet)|
| **catboost_regressor**  |Gradient boosting on decision trees - [`CatBoost`](https://catboost.ai/)|
| **random_forest_regressor**  |[`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)|
| **xgboost_regressor**  |[`XGBoostClassifier`](https://xgboost.readthedocs.io/en/stable/)|
| **neural_nets_regression**  |PyTorch-based neural net regressor.|
| **linear_regression**  |[`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)|


### Explainers
```python
from autoprognosis.plugins.explainers import Explainers

explainer = Explainers().get(<NAME>)
```
| Name | Description |
|--- | --- |
| **risk_effect_size**  | Feature importance using Cohen's distance between probabilities|
| **lime**  |[`Lime: Explaining the predictions of any machine learning classifier`](https://github.com/marcotcr/lime)|
| **symbolic_pursuit**  |[`Symbolic Pursuit`](Learning outside the black-box: at the pursuit of interpretable models)|
| **shap_permutation_sampler**  |[`SHAP Permutation Sampler`](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Permutation.html)|
| **kernel_shap**  |[`SHAP KernelExplainer`](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)|
| **invase**  |[`INVASE: Instance-wise Variable Selection`](https://github.com/vanderschaarlab/invase)|



### Uncertainty
```python
from autoprognosis.plugins.uncertainty import UncertaintyQuantification
model = UncertaintyQuantification().get(<NAME>)
```
| Name | Description |
|--- | --- |
| **cohort_explainer**  ||
| **conformal_prediction**  ||
| **jackknife**  ||


## :hammer: Test
After installing the library, the tests can be executed using `pytest`
```bash
$ pip install .[dev]
$ pytest -vxs -m "not slow"
```

## Citing
If you use this code, please cite the associated paper:

```
@misc{https://doi.org/10.48550/arxiv.2210.12090,
  doi = {10.48550/ARXIV.2210.12090},
  url = {https://arxiv.org/abs/2210.12090},
  author = {Imrie, Fergus and Cebere, Bogdan and McKinney, Eoin F. and van der Schaar, Mihaela},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {AutoPrognosis 2.0: Democratizing Diagnostic and Prognostic Modeling in Healthcare with Automated Machine Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## References
1. [AutoPrognosis: Automated Clinical Prognostic Modeling via Bayesian Optimization with Structured Kernel Learning](https://arxiv.org/abs/1802.07207)
2. [Prognostication and Risk Factors for Cystic Fibrosis via Automated Machine Learning](https://www.nature.com/articles/s41598-018-29523-2)
3. [Cardiovascular Disease Risk Prediction using Automated Machine Learning: A Prospective Study of 423,604 UK Biobank Participants](https://www.ncbi.nlm.nih.gov/pubmed/31091238)
