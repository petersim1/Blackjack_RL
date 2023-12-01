from typing import List, Union

import numpy as np
import pandas as pd

colorMap = {
    "S": "yellow",
    "H": "white",
    "D": "green",
    "Su": "grey",
    "Sp": "blue",
    "D/H": "green",
    "D/S": "green",
    "Su/H": "grey",
    "Su/S": "grey",
    "Y": "green",
    "N": "white"
}


def show_action_table(
    data: np.ndarray,
    xticks: List[Union[int, str]],
    yticks: List[Union[int, str]],
) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df.index = yticks
    df.columns = xticks
    df = df.style.applymap(lambda x: "background-color: %s; color: black" % colorMap[x])
    return df


__all__ = ["show_action_table"]
