
<h1 align="center">
  <br>
  <a href="https://www.vanderschaar-lab.com/"><img src="https://www.vanderschaar-lab.com/wp-content/uploads/2020/07/AutoML_Fig1_rev-2048x1199.png" alt="Adjutorium" width="400"></a>
  <br>
  Adjutorium
  <br>
</h1>

<h3 align="center">
  <br>
  A system for automating the design of predictive modeling pipelines tailored for clinical prognosis.
  <br>
</h3>

<div align="center">

[![Tests](https://github.com/vanderschaarlab/adjutorium-framework/actions/workflows/test.yml/badge.svg)](https://github.com/vanderschaarlab/adjutorium-framework/actions/workflows/test.yml)
[![Tutorials](https://github.com/vanderschaarlab/adjutorium-framework/actions/workflows/test_tutorials.yml/badge.svg)](https://github.com/vanderschaarlab/adjutorium-framework/actions/workflows/test_tutorials.yml)
[![Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://vanderschaarlab.slack.com/messages/general)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/vanderschaarlab/adjutorium-framewor/blob/main/LICENSE)


</div>

## Features

- :key: Automatically learns ensembles of pipelines for prediction or survival analysis.
- :cyclone: Easy to extend pluginable architecture.
- :zap: Survival analysis and Treatment effects models.
- :fire: Interpretability tools.

## Installation

#### Using pip

```bash
$ pip install -r requirements_dev.txt
$ pip install .
```

### Redis (Optional, but recommended)
Adjutorium can use Redis as a backend to improve the performance and quality of the searches.

For that, install the redis-server package following the steps described on the [official site](https://redis.io/topics/quickstart).

## Testing
After installing the library, the tests can be executed using `pytest`
```bash
$ pytest -vxsx -m "not slow"
```
## Using the library
More advanced use cases can be found on our [tutorials section](#tutorials).

#### Example: list available classifiers
```python
from adjutorium.plugins.prediction.classifiers import Classifiers
print(Classifiers().list())
```

#### Example for classification estimators studies
```python
from pathlib import Path

from sklearn.datasets import load_breast_cancer

from adjutorium.studies.classifiers import ClassifierStudy
from adjutorium.utils.serialization import load_model_from_file
from adjutorium.utils.tester import evaluate_estimator


X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

df = X.copy()
df["target"] = Y

workspace = Path("workspace")
study_name = "example"

study = ClassifierStudy(
    study_name=study_name,
    dataset=df,  # pandas DataFrame
    target="target",  # the label column in the dataset
    num_iter=2,  # how many trials to do for each candidate
    timeout=10,  # seconds
    classifiers=["logistic_regression", "lda", "qda"],
    workspace=workspace,
)

study.run()

output = workspace / study_name / "model.p"
model = load_model_from_file(output)

metrics = evaluate_estimator(model, X, Y)

print(f"model {model.name()} -> {metrics['clf']}")
```

#### Example: list available survival analysis estimators
```python
from adjutorium.plugins.prediction.risk_estimation import RiskEstimation
print(RiskEstimation().list())
```
### Example for survival analysis studies
```python
import os
from pathlib import Path

from lifelines.datasets import load_rossi

from adjutorium.studies.risk_estimation import RiskEstimationStudy
from adjutorium.utils.serialization import load_model_from_file
from adjutorium.utils.tester import evaluate_survival_estimator


rossi = load_rossi()

X = rossi.drop(["week", "arrest"], axis=1)
Y = rossi["arrest"]
T = rossi["week"]

eval_time_horizons = [
    int(T[Y.iloc[:] == 1].quantile(0.25)),
    int(T[Y.iloc[:] == 1].quantile(0.50)),
    int(T[Y.iloc[:] == 1].quantile(0.75)),
]

workspace = Path("workspace")
study_name = "example_risks"

study = RiskEstimationStudy(
    study_name=study_name,
    dataset=rossi,
    target="arrest",
    time_to_event="week",
    time_horizons=eval_time_horizons,
    num_iter=2,
    num_study_iter=1,
    timeout=10,
    risk_estimators=["cox_ph", "lognormal_aft", "loglogistic_aft"],
    workspace=workspace,
)

study.run()

output = workspace / study_name / "model.p"

if output.exists():
    model = load_model_from_file(output)

    metrics = evaluate_survival_estimator(model, X, T, Y, eval_time_horizons)

    print(f"Model {model.name()} score: {metrics['clf']}")
```
## Using the UI
1. Install and start [Redis](https://redis.io/topics/quickstart).
2. Install [Docker](https://docs.docker.com/get-started/).
3. Build the Adjutorium Docker image ```docker build -t adjutorium .```.
4. Start the Adjutorium server ```docker run --network host -it -p 8002:8002 adjutorium:latest```.
5. The interface will be available at [http://127.0.0.1:8002/](http://127.0.0.1:8002/).

## Tutorials

### Plugins
- [Imputation ](tutorials/plugins/tutorial_00_imputer_plugins.ipynb)
- [Preprocessing](tutorial_01_preprocessing_plugins.ipynb)
- [Classification](tutorials/plugins/tutorial_02_classification_plugins.ipynb)
- [Calibration](tutorials/plugins/tutorial_03_calibration_plugins.ipynb)
- [Pipelines](tutorials/plugins/tutorial_04_pipelines.ipynb)
- [Treatments](tutorials/plugins/tutorial_05_treatments.ipynb)
### AutoML
 - [Classification tasks](tutorials/automl/tutorial_00_classification_study.ipynb)
 - [Survival analysis](tutorials/automl/tutorial_01_survival_analysis_study.ipynb)

## References
1. [Adjutorium: Automated Clinical Prognostic Modeling via Bayesian Optimization with Structured Kernel Learning](https://arxiv.org/abs/1802.07207)
2. [Prognostication and Risk Factors for Cystic Fibrosis via Automated Machine Learning](https://www.nature.com/articles/s41598-018-29523-2)
3. [Cardiovascular Disease Risk Prediction using Automated Machine Learning: A Prospective Study of 423,604 UK Biobank Participants](https://www.ncbi.nlm.nih.gov/pubmed/31091238)

## Presentation

<div align="center">

[![Adjutorium: Automatic Prognostic Modelling](https://img.youtube.com/vi/d1uEATa0qIo/0.jpg)](https://www.youtube.com/watch?v=d1uEATa0qIo "Automatic Prognostic Modelling")

</div>

## License
[MIT License](https://github.com/vanderschaarlab/adjutorium-priv/blob/main/LICENSE)
