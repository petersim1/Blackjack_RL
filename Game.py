import numpy as np

'''
This module controls the blackjack gameplay.
It wraps the Player.py module, which represents each player and the house
'''

class Game :
    
    def __init__(self,playerModule,allowHardCodedCards=False,shrinkDeck=True,nDecks=6,ratioPenetrate=4/6,verbose=True) :
        
        '''
        Input :
            - playerModule : class , uninitialized Player module from Player.py
            - shrinkDeck : boolean , whether or not to remove selected cards from deck. If False, each card is drawn iid.
            - nDecks : number of decks to play with (default is 6, which is typical)
            - ratioPenetrate : ratio of cards that are playable (default is 2/3 of 6 decks). Only applicable when shrinkDeck == True.
            - verbose : boolean , whether to print when deck is cut / reshuffled.
            
            MUST call initRound() to start the round
        '''
        self.cardMap = {i:k for i,k in enumerate(list(range(2,11)) + ['J','Q','K','A'])}
        self.cardValues = {k:k for k in range(2,11)}
        for c in ['J','Q','K'] :
            self.cardValues[c] = 10
        self.cardValues['A'] = 1
        
        self.allowHardCodedCards = allowHardCodedCards # Whether or not to allow hardcoded cards as inputs (useful in training).
        self.shrinkDeck = shrinkDeck # if False, will randomly select cards uniformly, and deck won't run out.
        self.verbose = verbose
        self.nDecks = nDecks
        self.ratioPenetrate = ratioPenetrate
        self.nRoundsPlayed = 0
        self.resetDeckAfterRound = False
        self.roundInit = False
        
        self.player = playerModule
        
        self._initDeck()
            
    def _initDeck(self) :
        
        self.cards = np.array([self.nDecks*4]*len(self.cardMap))
            
        self.nCardsPlayed = 0 # In THIS deck.
        self.count = 0
        self.stopCard = int(self.nDecks*52*self.ratioPenetrate)
        
    def _initPlayers(self) :
        
        '''
        Initialize hands of players and house
        '''
        
        self.players = [self.player(wager,self.cardValues) for wager in self.wagers]
        self.house = self.player(0,self.cardValues)
        
    def _updateCount(self,card) :
        
        if self.cardValues[card] <=6 :
            self.count += 1
        if self.cardValues[card] >= 10 :
            self.count -= 1
            
    def _selectCard(self,updateCount=True,hardcodedCard=None) :
        
        if hardcodedCard is None :
            # add clipping to cater for hardcoded card values.
            ind = np.random.choice(list(self.cardMap.keys()),p=np.clip(self.cards,a_min=0,a_max=None)/np.clip(self.cards,a_min=0,a_max=None).sum())
        else :
            ind = list(self.cardMap.values()).index(hardcodedCard)
            if self.cards[ind] <= 0 :
                self.resetDeckAfterRound = True # on pairs, possible to go negative on number left of that card, but reset deck after depleted.
        
        card = self.cardMap[ind]
        if self.shrinkDeck :
            self.cards[ind] -= 1
            self.nCardsPlayed += 1
        
        if self.nCardsPlayed == self.stopCard :
            if self.verbose :
                print('Stop Card Hit, resetting deck after this round')
            self.resetDeckAfterRound = True
        
        if updateCount :
            self._updateCount(card)
        
        return card
        
    def initRound(self,wagers) :
        
        '''
        - Calls to reset the player / house cards
        - Increments the # of rounds played
        '''
        self.wagers = wagers
        
        self._initPlayers()
        
        self.nRoundsPlayed += 1
        self.roundInit = True
        self.houseBlackjack = False
        
        if self.resetDeckAfterRound : 
            if self.verbose :
                print('shuffling deck')
            self._initDeck()
            self.resetDeckAfterRound = False
        
    def resetGame(self) :
        
        '''
        Resets the entire game. ie game is over, reset module.
        '''
        
        self._initDeck()
        self.players = []
        self.house = None
        self.nRoundsPlayed = 0      
        
    def dealInit(self,hardcodedCards=None) :
        
        assert self.roundInit , 'Must initialize round before dealing'
        if hardcodedCards is not None :
            assert self.allowHardCodedCards, "Must initialize module with allowHardCodedCards=True to use this"

        for i in range(2) :
            for j,player in enumerate(self.players) :
                if hardcodedCards is None :
                    card = self._selectCard()
                else :
                    card = self._selectCard(hardcodedCard=hardcodedCards[0][j][i])
                player._dealCard(card)
            if hardcodedCards is None :
                card = self._selectCard(updateCount=(1-i)) #first card is shown, 2nd is hidden
            else :
                card = self._selectCard(updateCount=(1-i),hardcodedCard=hardcodedCards[1][i])
            self.house._dealCard(card)
        
        house,_,_,_ = self.house.getValue()
        if house == 21 : 
            self.houseBlackjack = True # If house has blackjack, don't accept bets (except insurance)
        
    def getHouseShow(self,showValue=False) :
        
        assert len(self.house.getCards()[0]) , 'House has not been dealt yet'
        
        card = self.house.getCards()[0][0]
        if showValue :
            return self.cardValues[card] if card != 'A' else 11
        return card
             
    def stepHouse(self) :
        
        house,_,_,_ = self.house.getValue()
        self._updateCount(self.house.cards[0][-1]) # 2nd card is now displayed, so adjust count.
        
        while house < 17 :
            card = self._selectCard()
            self.house._dealCard(card)
            house,_,_,_ = self.house.getValue()
            
    def stepPlayer(self,player,move) :
        
        n = player.getNumCardsDraw(move)
        
        cards = [self._selectCard() for _ in range(n)]
        
        player.step(move,cards)
        
    
    def getResults(self) :
        
        house,_,_,_ = self.house.getValue()
        
        players = []
        winnings = []
        for player in self.players :
            
            text,win = player.getResult(house,self.house.cards)
            players.append(text)
            winnings.append(win)
            
                
        return players,winnings