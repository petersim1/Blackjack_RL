from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(
    array: List[float],
    every: int,
    label: str,
    include_avg: bool = True,
    include_max: bool = False,
    **kwargs,
) -> None:
    plt.figure(figsize=(15, 4))
    plt.plot(np.arange(0, len(array)) * every, array, label=label)
    if include_avg:
        plt.plot(
            np.arange(0, len(array)) * every,
            np.cumsum(array) / np.arange(1, len(array) + 1),
            label="Rolling Avg.",
        )
    if include_max:
        plt.vlines(
            x=np.argmax(array) * every,
            ymin=min(array),
            ymax=max(array),
            color="k",
        )
    plt.title(kwargs["title"])
    plt.ylabel(kwargs["ylabel"])
    if include_avg:
        plt.legend()
    plt.show()


def plot_correctness(array, every) -> None:
    plt.figure(figsize=(15, 4))
    plt.title("Percent correct moves compared to baseline")
    plt.plot(np.arange(0, len(array)) * every, array)
    plt.plot(
        np.arange(0, len(array)) * every,
        np.cumsum(array) / np.arange(1, len(array) + 1),
        label="Rolling Avg.",
    )
    plt.legend()
    plt.show()


def plot_bar(datas, labels, alphas):
    unique_vals = np.unique(datas)

    for i, data in enumerate(datas):
        out = []
        values, freq = np.unique(data, return_counts=True)
        freq = freq / freq.sum()
        freq_dict = dict(zip(values, freq))

        for val in unique_vals:
            if val in freq_dict:
                out.append(freq_dict[val])
            else:
                out.append(0)

        plt.plot(unique_vals, out, label=labels[i], alpha=alphas[i])


def plot_mesh(axis, data, ranges, ticks=None):
    x, y = np.meshgrid(ranges[0], ranges[1])
    axis.plot_surface(
        x, y, data, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )
    axis.view_init(azim=-25)
    axis.set_xlabel("House Shows")
    axis.set_ylabel("Player Shows")
    axis.set_zlabel("Value")
    if ticks is not None:
        axis.set(yticks=ranges[0], yticklabels=ticks)


def clean_text(val1, val2=""):
    if val1 not in ["surrender", "split"]:
        val1 = val1[:1].title()
    else:
        val1 = val1[:2].title()
    if val2:
        if val2 not in ["surrender", "split"]:
            val2 = val2[:1].title()
        else:
            val2 = val2[:2].title()
        val = val1 + "/" + val2
    else:
        val = val1
    return val


def extract_best(values: dict, return_type: str = "string"):
    moves = ["hit", "stay", "double", "surrender"]

    dict_moves = {k: v for k, v in values.items() if k in moves}

    if return_type == "value":
        return max(dict_moves.values())

    max_val1 = max(dict_moves, key=dict_moves.get)
    max_val2 = ""
    if max_val1 in ["double", "surrender"]:
        moves = ["hit", "stay"]
        dict_moves = {k: v for k, v in values.items() if k in moves}
        max_val2 = max(dict_moves, key=dict_moves.get)

    return clean_text(max_val1, max_val2)


def generate_grid(
        q: dict,
        return_type: str = "string") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        hard = np.empty((16, 10), dtype="O")
        soft = np.empty((8, 10), dtype="O")
        split = np.empty((10, 10), dtype="O")
    else:
        hard = np.full((16, 10), np.nan)
        soft = np.full((8, 10), np.nan)
        split = np.full((10, 10), np.nan)

    for (player, house, useable_ace), vals in q.items():
        vals: dict
        can_split = (not player % 2) and ((not useable_ace) or (player == 12))

        if (not useable_ace) and (4 < player < 21):
            hard[20 - player, house - 2] = extract_best(vals, return_type=return_type)
        if useable_ace and (not can_split):
            soft[20 - player, house - 2] = extract_best(vals, return_type=return_type)
        if can_split:
            if return_type == "string":
                max_val = clean_text(max(vals, key=vals.get), "")
            else:
                max_val = max(vals.values())
            player_ind = 11 if useable_ace else int(player / 2)
            split[11 - player_ind, house - 2] = max_val

    return hard, soft, split
