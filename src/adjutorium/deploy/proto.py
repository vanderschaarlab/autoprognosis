# third party
from pydantic import BaseModel


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
