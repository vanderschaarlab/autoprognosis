# stdlib
import collections
import hashlib
import json
from typing import List

# third party
import numpy as np
import pandas as pd
from pymfe.mfe import MFE

# adjutorium absolute
import adjutorium.plugins.utils.cast as cast


class DatasetMetafeatures:
    def __init__(self, X: pd.DataFrame) -> None:
        X = cast.to_ndarray(X)

        mfe = MFE(groups=["default"])
        mfe.fit(X)
        ft = mfe.extract()

        self.features = collections.OrderedDict()
        for x, y in zip(ft[0], ft[1]):
            if np.isnan(y):
                y = 0
            self.features[x] = float(y)

    def hash(self) -> str:
        return hashlib.sha1(
            json.dumps(self.features, sort_keys=True).encode()
        ).hexdigest()

    def dict(self) -> dict:
        return self.features

    def array(self) -> List:
        return list(self.features.values())

    def distance(self, other: "DatasetMetafeatures") -> float:
        dist = 0
        for v1, v2 in zip(self.array(), other.array()):
            dist += abs(v1 - v2)

        return dist
