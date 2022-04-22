import numpy as np

def initQ(moves) :

    Q = {}
    for p in range(4,22) :
        for h in range(2,12) :
            if (21 > p > 11)   :
                for a in [True,False] :
                    Q[(p,h,a)] = {m:0 for m in moves}
            else :
                Q[(p,h,False)] = {m:0 for m in moves}
                
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

        playerShow,useableAce = player.getValue()
        nHand = player._getCurHand()

        policy = player.getValidMoves(houseShow)
        policy = [p for p in policy if p!='insurance']
        move = getBestAction(Q[(playerShow,houseShow,useableAce)],policy,epsilon)

        s_a_pairs[nHand].append((playerShow,houseShow,useableAce,move))

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
                        
            p,h,a,m = s_a_pairs[i][hand][j] # current state-action pair (player,house,useableAce,move)
            oldQ = Q[(p,h,a)][m] # Q value for current state-action pair
            
            r = w
            maxQ_p = 0
            if (j+1) < len(s_a_pairs[i][hand]) :
                p_p,h_p,a_p,_ = s_a_pairs[i][hand][j+1]
                maxQ_p = max(Q[(p_p,h_p,a_p)].values()) # get maximum Q value for s`
                r = 0
            
            Q[(p,h,a)][m] = oldQ + lr*(r + gamma*maxQ_p - oldQ)
            
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
                
                playerShow,useableAce = player.getValue()
                policy = player.getValidMoves(houseShow)
                policy = [p for p in policy if p!='insurance']
                move = getBestAction(Q[(playerShow,houseShow,useableAce)],policy,-1)

                blackjack.stepPlayer(player,move)

        blackjack.stepHouse() #play the house complete hand.

        _,playerWinnings = blackjack.getResults()


        for i,w in enumerate(playerWinnings) :

            rewards[i].append(w)

    return rewards