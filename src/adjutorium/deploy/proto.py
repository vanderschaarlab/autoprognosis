# third party
from pydantic import BaseModel


class StudyProto(BaseModel):
    id: str


class NewStudyProto(BaseModel):
    name: str
    type: str
    workspace: str
    dataset_path: str
    time_column: str
    target_column: str
    horizons: list
    imputers: list
    risk_estimators: list


class NewAppProto(BaseModel):
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


class AppProto(BaseModel):
    app_path: str


class StatisticsUpdateProto(BaseModel):
    id: str
    topic: str
    subtopic: str
    name: str
    aucroc: str
    cindex: str
    brier_score: str
    duration: float
    horizon: int


class StatisticsQueryProto(BaseModel):
    id: str
    horizons: list
