import numpy as np
import pandas as pd


def dfBestMove(array,moves,pairsSplit,colorBox=True) :

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