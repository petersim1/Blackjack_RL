from typing import List, Optional, Union

import numpy as np
import pandas as pd

colorMap = {
    "St": "yellow",
    "Hi": "white",
    "Do": "green",
    "Su": "grey",
    "Sp": "blue",
    "Do/Hi": "green",
    "Do/St": "green",
}


def show_action_table(
    data: np.ndarray,
    xticks: Optional[List[Union[int, str]]] = None,
    yticks: Optional[List[Union[int, str]]] = None,
) -> pd.DataFrame:
    x_range = np.where(data.any(axis=1))[0]
    y_range = np.where(data.any(axis=0))[0]

    data = data = data[x_range, :][:, y_range]

    df = pd.DataFrame(data)
    if xticks:
        df.index = xticks
    else:
        df.index = x_range
    if yticks:
        df.columns = yticks
    else:
        df.columns = y_range

    df = df.style.applymap(lambda x: "background-color: %s; color: black" % colorMap[x])

    return df


__all__ = ["show_action_table"]
