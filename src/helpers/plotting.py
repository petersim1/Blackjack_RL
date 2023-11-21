from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(array: List[float], every: int, label: str, include_max: bool=False) -> None:
    plt.figure(figsize=(15,4))
    plt.plot(
        np.arange(0,len(array))*every,
        array,
        label=label
    )
    plt.plot(
        np.arange(0,len(array))*every,
        np.cumsum(array) / np.arange(1,len(array)+1),
        label="Rolling Avg."
    )
    if include_max:
        plt.vlines(x=np.argmax(array)*every,ymin=min(array),ymax=max(array),color="k")
    plt.title("Q-Learning")
    plt.ylabel("Avg Reward at Evaluation")
    plt.legend()
    plt.show()


def plot_correctness(array, every) -> None:
    plt.figure(figsize=(15,4))
    plt.title("Percent correct moves compared to baseline")
    plt.plot(np.arange(0,len(array))*every, array)
    plt.show()


def plot_bar(datas, labels, alphas):

    unique_vals = np.unique(datas)

    for i,data in enumerate(datas):
        out = []
        values, freq = np.unique(data, return_counts=True)
        freq = freq / freq.sum()
        freq_dict = dict(zip(values,freq))

        for val in unique_vals:
            if val in freq_dict:
                out.append(freq_dict[val])
            else:
                out.append(0)

        plt.plot(unique_vals, out, label=labels[i], alpha=alphas[i])


def plot_mesh(axis, data, ranges, ticks=None):
    x,y = np.meshgrid(ranges[0], ranges[1])
    axis.plot_surface(x, y, data, rstride=1, cstride=1,cmap="viridis", edgecolor="none")
    axis.view_init(azim=-25)
    axis.set_xlabel("House Shows")
    axis.set_ylabel("Player Shows")
    axis.set_zlabel("Value")
    if ticks is not None:
        axis.set(yticks=ranges[0], yticklabels=ticks)


def extract_best(values: dict, return_type: str="string"):
    moves = ["hit", "stay", "double", "surrender"]

    dict_moves = {k:v for k,v in values.items() if k in moves}

    if return_type == "value":
        return max(dict_moves.values())
    
    max_val = max(dict_moves,key=dict_moves.get)
    if max_val in ["double", "surrender"]:
        moves = ["hit", "stay"]
        dict_moves = {k:v for k,v in values.items() if k in moves}
        max_val = max_val[:2].title() + "/" + max(dict_moves, key=dict_moves.get)[:2].title()
    else:
        max_val = max_val[:2].title()

    return max_val

def generate_grid(q: dict, return_type: str="string") -> np.ndarray:
    """
    takes a Q dict and a return_type
    if return_type == "string":
        return a str detailing the best move.
        if best move is double / surrender (when you can't split), also
        append best move for instance when you can't actually double / surrender
    if return_type == "value"
        simply return the maximum q value
    do this for each of hard total, soft total, can split.
    """
    assert return_type in ["string", "value"], "invalid return_type"
    if return_type == "string":
        fill = np.empty((3, 21 + 1, 11 + 1), dtype="O")
    else:
        fill = np.full((3, 21 + 1, 11 + 1), np.nan)

    for (player, house, useable_ace), vals in q.items():
        vals: dict
        can_split = not player % 2

        if not useable_ace:
            fill[0, player, house] = extract_best(vals, return_type=return_type)
        if useable_ace and (player != 12):
            fill[1, player, house] = extract_best(vals, return_type=return_type)
        if can_split and ((not useable_ace) or (player == 12)):
            if return_type == "string":
                max_val = max(vals, key=vals.get)[:2].title()
            else:
                max_val = max(vals.values())
            player_ind = 11 if useable_ace else int(player / 2)
            fill[2, player_ind, house] = max_val
    
    return fill