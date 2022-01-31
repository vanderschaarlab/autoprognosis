# adjutorium absolute
from adjutorium.utils.pip import install

for retry in range(2):
    try:
        # third party
        from pydantic import BaseModel

        break
    except ImportError:
        depends = ["pydantic"]
        install(depends)


class NewRiskEstimationAppProto(BaseModel):
    name: str
    type: str
    dataset_path: str
    model_path: str
    time_column: str
    target_column: str
    horizons: list
    explainers: list
    imputers: list
    plot_alternatives: list


class NewClassificationAppProto(BaseModel):
    name: str
    type: str
    dataset_path: str
    model_path: str
    target_column: str
    explainers: list
    imputers: list
    plot_alternatives: list
