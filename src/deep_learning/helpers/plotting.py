from __future__ import annotations
import numpy as np
from typing import List, Tuple, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors.
    from src.deep_learning.modules import Net

def interp(data, interpol=1):

    n,m = data.shape
    data_inter = np.zeros((
        n + (n-1)*(interpol-1),
        m + (m-1)*(interpol-1)
    ))
    x_i = np.linspace(0,1,interpol+1)
    y_i = np.linspace(0,1,interpol+1)

    for row in range(data.shape[0]-1):
        for col in range(data.shape[1]-1):
            d = data[row:(row+2),col:(col+2)]
            inter_d = np.array([1-x_i, x_i]).T.dot(d).dot(np.array([[1-y_i],[y_i]])[:,0,:])
            data_inter[row*interpol:(row+1)*interpol+1, col*interpol:(col+1)*interpol+1] = inter_d

    return data_inter


def plot_mesh(axis, data, ranges, interpolate=1, ticks=None, zlims=[], **kwargs):

    data_interp = interp(data, interpolate)

    x_r = np.linspace(min(ranges[0]), max(ranges[0]), data_interp.shape[1])
    y_r = np.linspace(min(ranges[1]), max(ranges[1]), data_interp.shape[0])


    x,y = np.meshgrid(x_r, y_r)
    axis.plot_surface(x, y, data_interp, rstride=1, cstride=1,cmap="viridis", edgecolor="none", **kwargs)
    # axis.plot_surface(x, y, data)
    axis.view_init(azim=-25)
    axis.set_xlabel("House Shows")
    axis.set_ylabel("Player Shows")
    axis.set_zlabel("Value")
    if ticks is not None:
        axis.set(yticks=ranges[0], yticklabels=ticks)

    if zlims:
        axis.set_zlim(*zlims)


def gen_all_states(include_count: bool=False, true_count: float=0):
    all_states = []

    for player in range(4, 22):
        for house in range(2, 12):
            for ace in [-1, 1]:
                if (player < 12) and (ace > 0): continue
                if (player == 21) and (ace > 0): continue
                for split in [-1, 1]:
                    if (player == 4) and (split < 0): continue
                    if (player == 12) and (split < 0) and (ace > 0): continue
                    if (player > 12) and (not player % 2) and (split > 0) and (ace > 0): continue
                    if (player % 2) and (split > 0): continue
                    for double in [-1, 1]:
                        if (split > 0) and (double < 0): continue
                        if (player == 5) and (double < 0): continue
                        if include_count:
                            all_states.append((player, house, ace, split, double, true_count))
                        else:
                            all_states.append((player, house, ace, split, double))

    return all_states


def fill_value_tensor(
        model: type[Net],
        no_ace_no_split: List[Tuple],
        yes_ace_no_split: List[Tuple],
        yes_split: List[Tuple]
):
    model.eval()
    value_det = np.zeros((3, 21+1, 11+1))

    _, q_values_max, _ = model.act(
        torch.tensor(no_ace_no_split, dtype=torch.float32),
        method="argmax",
        avail_actions=[["stay", "hit", "double"] for _ in no_ace_no_split]
    )
    for i,q in enumerate(q_values_max):
        value_det[0, no_ace_no_split[i][0], no_ace_no_split[i][1]] = q


    _, q_values_max, _ = model.act(
        torch.tensor(yes_ace_no_split, dtype=torch.float32),
        method="argmax",
        avail_actions=[["stay", "hit", "double"] for _ in yes_ace_no_split]
    )
    for i,q in enumerate(q_values_max):
        value_det[1, yes_ace_no_split[i][0], yes_ace_no_split[i][1]] = q


    _, q_values_max, _ = model.act(
        torch.tensor(yes_split, dtype=torch.float32),
        method="argmax",
        avail_actions=[["stay", "hit", "double", "split"] for _ in yes_split]
    )
    for i,q in enumerate(q_values_max):
        player_ind = int(yes_split[i][0] / 2)
        if player_ind == 6:
            if yes_split[i][2]:
                player_ind = 11
        value_det[2, player_ind, yes_split[i][1]] = q

    return value_det

__all__ = ["plot_mesh", "gen_all_states", "fill_value_tensor"]