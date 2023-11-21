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


def generate_mesh(q: object) -> np.ndarray:
    value_det = np.zeros((3, 21+1, 11+1))

    for (player, house, useable_ace, can_split), vals in q.items():
        moves = ["stay", "hit", "double"]
        if can_split:
            moves.append("split")

        dict_moves = {k:v for k,v in vals.items() if k in moves}
        if not can_split:
            value_det[int(useable_ace), player, house] = max(dict_moves.values())
        else:
            player_ind = int(player / 2)
            if useable_ace:
                if player == 12:
                    player_ind = 11
                else:
                    continue
            value_det[2, player_ind, house] =  max(dict_moves.values())

    return value_det

def hard_totals(q: dict):
    # assume no useable ace, no ability to split
    fill = np.empty((21 + 1, 11 + 1), dtype="O")

    for (player, house, useable_ace), vals in q.items():
        if useable_ace: continue
        vals: dict
        # first assume we can double / surrender.
        moves = ["hit", "stay", "double", "surrender"]

        dict_moves = {k:v for k,v in vals.items() if k in moves}
        max_val = max(dict_moves,key=dict_moves.get)

        if max_val in ["double", "surrender"]:
            moves = ["hit", "stay"]
            dict_moves = {k:v for k,v in vals.items() if k in moves}
            max_val = max_val[:2].title() + "/" + max(dict_moves,key=dict_moves .get)[:2].title()
        else:
            max_val = max_val[:2].title()
        
        fill[player, house] = max_val
    
    return fill

def soft_totals(q: dict):

    fill = np.empty((21 + 1, 11 + 1), dtype="O")

    for (player, house, useable_ace), vals in q.items():
        if not useable_ace: continue
        if player == 12: continue # specifical designation for (A,A)
        vals: dict
        moves = ["hit", "stay", "double", "surrender"]

        dict_moves = {k:v for k,v in vals.items() if k in moves}
        max_val = max(dict_moves,key=dict_moves.get)

        if max_val in ["double", "surrender"]:
            moves = ["hit", "stay"]
            dict_moves = {k:v for k,v in vals.items() if k in moves}
            max_val = max_val[:2].title() + "/" + max(dict_moves,key=dict_moves .get)[:2].title()
        else:
            max_val = max_val[:2].title()
        
        fill[player, house] = max_val
    return fill

def splits(q: dict):
    
    fill = np.empty((21 + 1, 11 + 1), dtype="O")

    for (player, house, useable_ace), vals in q.items():
        can_split = not player % 2
        if not can_split: continue
        if (player != 12) and useable_ace: continue # specific designation for (A,A) / (6,6)
        vals: dict

        max_val = max(vals, key=vals.get)
        # val = "Y" if max_val == "split" else "N"

        player_ind = 11 if useable_ace else int(player / 2)
        fill[player_ind, house] = max_val[:2].title()
    return fill

def generate_grid(q: dict) -> Tuple[np.ndarray, np.ndarray]:

    fill = np.empty((3, 21+1, 11+1), dtype="O")

    fill[0] = hard_totals(q)
    fill[1] = soft_totals(q)
    fill[2] = splits(q)

    return fill