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

    for (player, house, useable_ace), vals in q.items():
        dict_no_split = {k:v for k,v in vals.items() if k != "split"}
        value_det[int(useable_ace), player, house] = max(dict_no_split.values())

        if not player % 2:
            player_ind = int(player / 2)
            if useable_ace:
                player_ind = 11
            value_det[2, player_ind, house] = max(vals.values())

    return value_det


def generate_grid(q: object) -> Tuple[np.ndarray, np.ndarray]:

    fill = np.empty((3, 21+1, 11+1), dtype="O")

    for (player, house, useable_ace), vals in q.items():

        dict_no_split = {k:v for k,v in vals.items() if k != "split"}

        max_val = max(dict_no_split,key=dict_no_split.get)
        if max_val == "double":
            dict_no_double = {k:v for k,v in dict_no_split.items() if k != "double"}
            max_val = max_val[:2].title() + "/" + max(dict_no_double,key=dict_no_double.get)[:2].title()
        else:
            max_val = max_val[:2].title()

        fill[int(useable_ace), player, house] = max_val

        if not player % 2:
            player_ind = int(player / 2)
            if useable_ace:
                if player == 12:
                    player_ind = 11
                else:
                    continue

            max_val = max(vals,key=vals.get)
            if max_val == "double":
                dict_no_double = {k:v for k,v in vals.items() if k != "double"}
                max_val = max_val[:2].title() + "/" + max(dict_no_double,key=dict_no_double.get)[:2].title()
            else:
                max_val = max_val[:2].title()
            
            fill[2, player_ind, house] = max_val

    return fill