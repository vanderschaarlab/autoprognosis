# stdlib
from typing import Callable, Optional

from pydantic import BaseModel


class BaseAppProto(BaseModel):
    name: str
    type: str
    dataset_path: str
    model_path: str
    explainers: list
    imputers: list
    plot_alternatives: list


class NewRiskEstimationAppProto(BaseAppProto):
    time_column: str
    target_column: str
    horizons: list
    comparative_models: list
    extras_cbk: Optional[Callable]
    auth: bool = False


class NewClassificationAppProto(BaseAppProto):
    target_column: str
