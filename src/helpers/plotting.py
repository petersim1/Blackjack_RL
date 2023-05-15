from typing import List
import numpy as np
import pandas as pd
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
    plt.title("Average Max Q of Randomly Sampled States")
    plt.ylabel("Percent correct moves compared to baseline")
    plt.plot(np.arange(0,len(array))*every, array)
    plt.show()


def plot_hist_to_line(data, label, alpha, **kwargs):
    a,b = np.histogram(data, **kwargs)
    plt.plot([b[i-1:i+1].mean() for i in range(1,b.shape[0])],a, label=label, alpha=alpha)


def plot_mesh(axis, data, ranges, ticks=None):
    x,y = np.meshgrid(ranges[0], ranges[1])
    axis.plot_surface(x, y, data, rstride=1, cstride=1,cmap="viridis", edgecolor="none")
    axis.view_init(azim=-20)
    axis.set_xlabel("House Shows")
    axis.set_ylabel("Player Shows")
    axis.set_zlabel("Value")
    if ticks is not None:
        axis.set(yticks=ranges[0], yticklabels=ticks)


def df_best_move(array,moves,pairsSplit,colorBox=True) :

    bestMove = np.empty((3,21+1,11+1),dtype='O')
    pairsTicks = [k for k in pairsSplit.keys() if k[0] not in ['J','Q','K']]

    # In this first bucket, you don't have the chance to split.
    indsSearch = [i for i,v in enumerate(moves) if v!='split']
    movesSearch = [v for v in moves if v!='split']
    for ace in [0,1] :
        for i in range(array.shape[-2]-len(pairsTicks)) :
            for j in range(array.shape[-1]) :
                best = movesSearch[np.argmax(array[indsSearch,ace,i,j])]
                bestMove[ace][i][j] = best[:2].title()

    for i in range(array.shape[-2]-len(pairsTicks),array.shape[-2]) :
        for j in range(array.shape[-1]) :
            if array[:,0,i,j].sum() == 0 :
                best = moves[np.argmax(array[:,1,i,j])]
            else :
                best = moves[np.argmax(array[:,0,i,j])]
            bestMove[-1][i-22][j] = best[:2].title()

    colorMap = {'St':'blue','Hi':'green','Do':'red','Su':'grey','Sp':'yellow'}

    def color(val) :
        return 'background-color: %s' % colorMap[val]

    colsShow = [col for col in range(2,12) if col>1]
    noAce = pd.DataFrame(bestMove[0])
    noAce.columns = list(range(12))
    noAce = noAce.iloc[5:][colsShow]

    colsShow = [col for col in range(2,12) if col>1]
    yesAce = pd.DataFrame(bestMove[1])
    yesAce.columns = list(range(12))
    yesAce = yesAce.iloc[13:][colsShow]

    colsShow = [col for col in range(2,12) if col>1]
    canSplit = pd.DataFrame(bestMove[2])
    canSplit.columns = list(range(12))
    canSplit = canSplit.iloc[:10][colsShow]
    canSplit.index = pairsTicks
    
    if colorBox :
        noAce = noAce.style.applymap(color)
        yesAce = yesAce.style.applymap(color)
        canSplit = canSplit.style.applymap(color)

    return noAce,yesAce,canSplit

__all__ = ["dfBestMove"]