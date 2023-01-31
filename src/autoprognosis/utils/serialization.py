# stdlib
from pathlib import Path
from typing import Any, Union

# third party
import cloudpickle
import pandas as pd

# autoprognosis absolute
from autoprognosis.version import MAJOR_VERSION


def _add_version(obj: Any) -> Any:
    obj._serde_version = MAJOR_VERSION
    return obj


def _check_version(obj: Any) -> Any:
    local_version = obj._serde_version

    if not hasattr(obj, "_serde_version"):
        raise RuntimeError("Missing serialization version")

    if local_version != MAJOR_VERSION:
        raise ValueError(
            f"Serialized object mismatch. Current major version is {MAJOR_VERSION}, but the serialized object has version {local_version}."
        )


def save(obj: Any) -> bytes:
    obj = _add_version(obj)
    return cloudpickle.dumps(obj)


def load(buff: bytes) -> Any:
    obj = cloudpickle.loads(buff)
    _check_version(obj)
    return obj


def save_model(obj: Any) -> bytes:
    return save(obj)


def load_model(buff: bytes) -> Any:
    return load(buff)


def save_to_file(path: Union[str, Path], obj: Any) -> Any:
    obj = _add_version(obj)
    with open(path, "wb") as f:
        return cloudpickle.dump(obj, f)


def load_from_file(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        obj = cloudpickle.load(f)
        _check_version(obj)
        return obj


def save_model_to_file(path: Union[str, Path], obj: Any) -> Any:
    return save_to_file(path, obj)


def load_model_from_file(path: Union[str, Path]) -> Any:
    return load_from_file(path)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Dataframe hashing, used for caching/backups"""
    df.columns = df.columns.astype(str)
    cols = sorted(list(df.columns))
    return str(abs(pd.util.hash_pandas_object(df[cols].fillna(0)).sum()))
