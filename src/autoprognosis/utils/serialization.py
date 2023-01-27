# stdlib
from pathlib import Path
from typing import Any, Union

# third party
import cloudpickle
import pandas as pd


def save(model: Any) -> bytes:
    return cloudpickle.dumps(model)


def load(buff: bytes) -> Any:
    return cloudpickle.loads(buff)


def save_model(model: Any) -> bytes:
    return cloudpickle.dumps(model)


def load_model(buff: bytes) -> Any:
    return cloudpickle.loads(buff)


def save_to_file(path: Union[str, Path], model: Any) -> Any:
    with open(path, "wb") as f:
        return cloudpickle.dump(model, f)


def load_from_file(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def save_model_to_file(path: Union[str, Path], model: Any) -> Any:
    return save_to_file(path, model)


def load_model_from_file(path: Union[str, Path]) -> Any:
    return load_from_file(path)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Dataframe hashing, used for caching/backups"""
    df.columns = df.columns.astype(str)
    cols = sorted(list(df.columns))
    return str(abs(pd.util.hash_pandas_object(df[cols].fillna(0)).sum()))
