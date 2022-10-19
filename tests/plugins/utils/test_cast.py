# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins.utils.cast import to_dataframe


def test_cast_to_dataframe() -> None:
    simple_list = [[1, 2, 3]]

    cast = to_dataframe(simple_list)
    assert isinstance(cast, pd.DataFrame)

    cast = to_dataframe(pd.DataFrame(simple_list))
    assert isinstance(cast, pd.DataFrame)

    cast = to_dataframe(np.array(simple_list))
    assert isinstance(cast, pd.DataFrame)
