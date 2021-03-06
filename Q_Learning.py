import numpy as np

def initQ(moves,allCards) :

    '''
    I initially didn't have a way to tease out splittable hands vs. non-splittable hands.
    Also, pair of 6's has the same total of pair of A's, which I wasn't able to differentiate.
    I need this more specific Q value dictionary to be able to differentiate better.
    '''
    
    movesNoSplit = [m for m in moves if m!='split']

    Q = {
        'canSplit':{},
        'noSplit':{}
    }
    
    for p in range(5,22) :
        
        for h in range(2,12) :
            if (21 > p > 11) :
                for a in [True,False] :
                    Q['noSplit'][(p,h,a)] = {m:0 for m in movesNoSplit}
            else :
                Q['noSplit'][(p,h,False)] = {m:0 for m in movesNoSplit}
    
    for c in allCards :
        if c in ['J','Q','K'] :
            continue
        for h in range(2,12) :
            a = False if c!='A' else True
            Q['canSplit'][(c,h,a)] = {m:0 for m in moves}

    return Q

def getBestAction(state,policy,epsilon) :
    
    n = np.random.rand()
    if n < epsilon :
        move = np.random.choice(policy)
    else :
        qDict = {k:v for k,v in state.items() if k in policy}
        bestMove = [k for k,v in qDict.items() if v==max(list(qDict.values()))]
        move = np.random.choice(bestMove)
    
    return move


def genEpisode(blackjack,iPlayer,Q,epsilon) :
    
    '''
    Inputs :
        - blackjack : module of blackjack gameplay (will contain the state)
        - Q : q values for the state-action pairs
        - epsilon : e-greedy hyperparameter. Explore vs. exploit.
        
    Outputs : 
        - Updated blackjack module
    
    Returns :
        - s_a_pairs : state-action pairs
    '''
    
    s_a_pairs = [[]]
    
    player = blackjack.players[iPlayer]
    houseShow = blackjack.getHouseShow(showValue=True)

    while not player.isDone() :

        playerShow,canSplit,useableAce,card1 = player.getValue()
        nHand = player._getCurHand()

        policy = player.getValidMoves(houseShow)
        policy = [p for p in policy if p!='insurance']
        
        if canSplit :
            move = getBestAction(Q['canSplit'][(card1,houseShow,useableAce)],policy,epsilon)
        else :
            move = getBestAction(Q['noSplit'][(playerShow,houseShow,useableAce)],policy,epsilon)

        s_a_pairs[nHand].append((playerShow,houseShow,useableAce,canSplit,card1,move))

        if move == 'split' :
            s_a_pairs.append(s_a_pairs[nHand].copy())

        blackjack.stepPlayer(player,move)
        
    return s_a_pairs

def learnPolicy(blackjack,Q,nPlayers,epsilon,gamma,lr) :
    
    '''
    epsilon : e-greedy hyperparameter
    gamma : decay factor, which I use to discount rewards for earlier moves in a round
    lr : learning rate to update Q function
    
    returns :
        learned Q function
    '''

    s_a_pairs = []

    for i in range(nPlayers) :

        s_a_pairs.append(genEpisode(blackjack,i,Q,epsilon))

    blackjack.stepHouse() #play the house complete hand.

    _,playerWinnings = blackjack.getResults()


    for i,w in enumerate(playerWinnings) :

        j = 0
        hand = 0
        while hand < len(s_a_pairs[i]) :
            
            # current state-action pair 
            # (player,house,useableAce,canSplit,card_1,move)
            p,h,a,s,c1,m = s_a_pairs[i][hand][j]
            if s :
                oldQ = Q['canSplit'][(c1,h,a)][m] # Q value for current state-action pair if canSplit
            else :
                oldQ = Q['noSplit'][(p,h,a)][m] # Q value for current state-action pair if cannotSplit
            
            r = w
            maxQ_p = 0
            if (j+1) < len(s_a_pairs[i][hand]) :
                p_p,h_p,a_p,s_p,c1_p,_ = s_a_pairs[i][hand][j+1]
                # get maximum Q value for s`
                if s_p:
                    maxQ_p = max(Q['canSplit'][(c1_p,h_p,a_p)].values())
                else :
                    maxQ_p = max(Q['noSplit'][(p_p,h_p,a_p)].values())
                r = 0
            
            if s :
                Q['canSplit'][(c1,h,a)][m] = oldQ + lr*(r + gamma*maxQ_p - oldQ)
            else :
                Q['noSplit'][(p,h,a)][m] = oldQ + lr*(r + gamma*maxQ_p - oldQ)
            
            if j < len(s_a_pairs[i][hand])-1 :
                j += 1 # move to next player
            else : 
                hand += 1 # move to next hand for a player
                j = 0

def evaluatePolicy(blackjack,Q,wagers,nRounds) :
    
    rewards = [[] for _ in wagers]
        
    for r in range(nRounds) :
        blackjack.initRound(wagers) # must call this before dealing a round

        blackjack.dealInit() # initial deal, before players decide what to do.
        houseShow = blackjack.getHouseShow(showValue=True)  

        for i,player in enumerate(blackjack.players) :

            while not player.isDone() :
                
                playerShow,canSplit,useableAce,card1 = player.getValue()
                policy = player.getValidMoves(houseShow)
                policy = [p for p in policy if p!='insurance']
                if canSplit:
                    move = getBestAction(Q['canSplit'][(card1,houseShow,useableAce)],policy,-1)
                else :
                    move = getBestAction(Q['noSplit'][(playerShow,houseShow,useableAce)],policy,-1)

                blackjack.stepPlayer(player,move)

        blackjack.stepHouse() #play the house complete hand.

        _,playerWinnings = blackjack.getResults()


        for i,w in enumerate(playerWinnings) :

            rewards[i].append(w)

    return rewards