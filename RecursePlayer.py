import numpy as np


# I took away any mentions of Insurance, since it's a side bet.

def getReward(moves,house) :
    
    reward = 0
    
    if moves[-1][1] == 'surrender' :
        return reward - .5
    
    if moves[-1][-1] > 21 :
        return reward -1 * (2 if moves[-1][1] == 'double' else 1)
    
    if house > 21 :
        return reward + 1 * (2 if moves[-1][1] == 'double' else 1)
        
    if moves[-1][-1] == house :
        return reward + 0
    
    if moves[-1][-1] > house :
        return reward + 1 * (2 if moves[-1][1] == 'double' else 1)
    
    if moves[-1][-1] < house :
        return reward - 1 * (2 if moves[-1][1] == 'double' else 1)
        
    return reward

def selectCard(cardMap) :
        
    ind = np.random.choice(list(cardMap.keys()))
    card = cardMap[ind]

    return card

def getValue(cards,cardValues) :
    
    useableAce = False
    total = sum([cardValues[card] for card in cards])

    if cards.count('A') :
        if total <= 11 :
            total += 10
            useableAce = True
            
    return total,useableAce

def dealHouse(card,cardMap,cardValues) :
    
    houseCards = [card]    
    initVal,_ = getValue(houseCards,cardValues)
    val = initVal
    
    while val < 17 :
        ind = np.random.choice(list(cardMap.keys()))
        card = cardMap[ind]
        houseCards.append(card)
        val,_ = getValue(houseCards,cardValues)
        
    return houseCards,val,initVal


def recursePlayer(cards,house,cardMap,cardValues) :

    def recursion(series,cards,move,houseShow) :

        val,useableAce = getValue(cards,cardValues)
        
        if move in ['stay','double','surrender'] :
            seriesOut.append(series)
            return
        
        if val >= 21 :
            seriesOut.append(series)
            return
        
        pairSplit = []

        if move == '' :
            if cards[0]==cards[1] :
                pairSplit = [cards[0],cards[1]]
                if houseShow == 'A' :
                    policy = ['stay','hit','double','split','surrender']
                else :
                    policy = ['stay','hit','double','split','surrender']
            else :
                if houseShow == 'A' :
                    policy = ['stay','hit','double','surrender']
                else :
                    policy = ['stay','hit','double','surrender']
        elif move == 'split' :
            if len(cards)==2 :
                if cards[0]==cards[1] :
                    pairSplit = [cards[0],cards[1]]
                    policy = ['stay','hit','double','split']
                else :
                    policy = ['stay','hit','double']
            else :
                policy = ['stay','hit']
        elif move == 'insurance' :
            if cards[0]==cards[1] :
                pairSplit = [cards[0],cards[1]]
                policy = ['stay','hit','double','split','surrender']
            else :
                policy = ['stay','hit','double','surrender']
        else :
            policy = ['stay','hit']
                
        

        for move in policy :
            if move == 'split' :
                for _ in [0,1] :
                    card = selectCard(cardMap)
                    nextVal,_ = getValue([cards[0]]+[card],cardValues)
                    recursion(series + [[val,move,useableAce,pairSplit,nextVal]],[cards[0]]+[card],move,houseShow)
            elif move in ['stay','surrender'] :
                recursion(series + [[val,move,useableAce,pairSplit,val]],cards,move,houseShow)
            else :
                card = selectCard(cardMap)
                nextVal,_ = getValue(cards + [card],cardValues)
                recursion(series+[[val,move,useableAce,pairSplit,nextVal]],cards + [card],move,houseShow)

    seriesOut = []
    recursion([],cards,'',house)

    return seriesOut