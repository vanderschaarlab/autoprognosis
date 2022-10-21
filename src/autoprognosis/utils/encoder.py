# stdlib
import json
from typing import Any

# third party
import numpy as np


class numpy_encoder(json.JSONEncoder):
    """Helper for encoding jsons"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(numpy_encoder, self).default(obj)
