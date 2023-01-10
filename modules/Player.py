from constants import const_rules_common, const_values

'''
This module is to be used as an individual blackjack player
It is intended to be wrapped into the Game.py Module, which controls the blackjack game
'''


class Player: 
    
    '''
    Inputs : 
        - wager : int , initial wager for the player
        - cardValues : dict , represents card values for each card. Should be inhereted from Game.py Module.
    '''

    cardValues = const_values
    rules_common = const_rules_common
    
    def __init__(self,wager, rules={}) :
        
        self.cards = [[]]
        
        self.baseWager = wager
        self.wager = [wager]

        self.rules = self.rules_common
        for k,v in rules.items() :
            if k in self.rules_common : 
                assert isinstance(v,bool), "Must have a boolean in rule object."
                self.rules[k] = v
        
        self.complete = [0]
        self.insured = 0
        self.surrendered = 0
        self.aces_split = False
    
    def _getCurHand(self) :
        
        return self.complete.index(0) if 0 in self.complete else None
    
    def _allComplete(self) :
        
        return not self.complete.count(0)
    
    def _dealCard(self,card) :
        
        iHand = self._getCurHand()
        
        self.cards[iHand].append(card)

        if self._getValueCards(self.cards[iHand])[0] >= 21 :
            self.complete[iHand] = 1        
    
    def _split(self,cards) :
        
        iHand = self._getCurHand()
        
        card = self.cards[iHand].pop(-1)
        self.cards.insert(iHand+1,[card])
        self.wager.insert(iHand+1,self.baseWager)
        self.complete.insert(iHand+1,0)
        
        self.cards[iHand].append(cards[0])
        self.cards[iHand+1].append(cards[1])
    
    def _getValueCards(self,cards) :
        
        useableAce = False
        
        total = sum([self.cardValues[card] for card in cards])
    
        if cards.count('A') :
            if total <= 11 :
                total += 10
                useableAce = True

        return total,useableAce
    
    def getValue(self) :
        
        iHand = self._getCurHand()
        n = len(self.cards[iHand])

        canSplit = (n==2) & (self.cards[iHand][0] == self.cards[iHand][1])
        useableAce = False
        c1 = None
        if canSplit :
            c1 = self.cards[iHand][0]
            if c1 in ['J','Q','K'] :
                c1 = 10
        
        total = sum([self.cardValues[card] for card in self.cards[iHand]])
    
        if self.cards[iHand].count('A') :
            if total <= 11 :
                total += 10
                useableAce = True if total < 21 else False

        return total,canSplit,useableAce,c1
        
         
    def getValidMoves(self,houseShow) :
        
        possibleMoves = []
        
        iHand = self._getCurHand()
        if iHand is None :
            return possibleMoves
        
        val,_,_,_ = self.getValue()
        
        nHands = len(self.cards)
        n = len(self.cards[iHand])
        
        canHit = (not self.aces_split) | self.rules["hitAfterSplitAces"]
        canStay = (not self.aces_split) | self.rules["hitAfterSplitAces"]
        canSplit = (n==2) & (self.cards[iHand][0] == self.cards[iHand][1])
        canInsure = (houseShow=='A') & (n==2) & (nHands==1) & (not self.insured)
        canSurrender = (n==2) & (nHands==1) & (self.rules["allowLateSurrender"])
        canDouble = (n==2) & (((nHands > 1) & self.rules["doubleAfterSplit"]) | (nHands == 1)) & canHit
                
        if val < 21 :
            if canStay: possibleMoves.append("stay")
            if canHit : possibleMoves.append("hit")
            if canSplit : possibleMoves.append('split')
            if canInsure : possibleMoves.append('insurance')
            if canSurrender : possibleMoves.append('surrender')
            if canDouble : possibleMoves.append('double')
        if val == 21 :
            possibleMoves.append("stay")
        
        return possibleMoves
    
    def getNumCardsDraw(self,move) :
        
        if move in ['hit','double'] :
            return 1
        if move == 'split' :
            return 2
        
        return 0
    
    def step(self,move,cardsGive=[]) :
        
        assert not self._allComplete() , 'Player cannot move anymore!'
        assert len(cardsGive) == self.getNumCardsDraw(move) , 'Must provide proper # of cards!'
        
        iHand = self._getCurHand()

        if move == 'hit' :
            self._dealCard(cardsGive[0])
                
        if move == 'stay' :
            self.complete[iHand] = 1

        if move == 'double' :
            self.wager[iHand] *= 2
            self._dealCard(cardsGive[0])
            self.complete[iHand] = 1

        if move == 'insurance' :
            self.insured = 1

        if move == 'split' :
            if (self.cards[iHand][0] == "A") & (self.cards[iHand][1] == "A") : self.aces_split = True
            self._split(cardsGive)
            if (not self.rules["hitAfterSplitAces"]) and self.aces_split :
                if cardsGive[0] != "A" : self.complete[iHand] = 1
                if cardsGive[1] != "A" : self.complete[iHand+1] = 1


        if move == 'surrender' :
            self.surrendered = 1
            self.complete[iHand] = 1
            
    def getResult(self,houseValue,houseCards) :
        
        houseIsBlackjack = (houseValue==21) & (len(houseCards)==2)
        blackjackPayout = 1.5 if not self.rules["reducedBlackjackPayout"] else 1.2
        
        text = []
        winnings = 0
        
        if self.insured :
            if houseIsBlackjack : # insurance pays out 2:1
                winnings += self.baseWager 
            else :
                winnings -= self.baseWager/2
        
        if self.surrendered :
            return [['surrender'],winnings-self.baseWager/2]
        
        for i,cards in enumerate(self.cards) :
            val,_ = self._getValueCards(cards)
            
            # 21 after a split is not natural blackjack. It's just 21, even on first two cards.
            isBlackjack = (val==21) & (len(cards)==2) & (len(self.cards) == 1)
            
            if val > 21 :
                text.append('bust')
                winnings -= self.wager[i]
            
            if val == 21 :
                if isBlackjack :
                    if houseIsBlackjack :
                        text.append('push')
                    else :
                        text.append('blackjack')
                        winnings += self.wager[i]*blackjackPayout
                else :
                    if houseIsBlackjack :
                        text.append('loss')
                        winnings -= self.wager[i]
                    else :
                        if houseValue == 21 :
                            text.append('push')
                        else :
                            if (self.rules["pushDealer22"] and (houseValue == 22)) : 
                                text.append("push")
                            else :
                                text.append('win')
                                winnings += self.wager[i]

            if val < 21 :
                if houseValue > 21 :
                    if (self.rules["pushDealer22"] and (houseValue == 22)) : 
                        text.append("push")
                    else :
                        text.append('win')
                        winnings += self.wager[i]
                else :
                    if val > houseValue :
                        text.append('win')
                        winnings += self.wager[i]
                    elif val < houseValue :
                        text.append('loss')
                        winnings -= self.wager[i]
                    else :
                        text.append('push')
        
        return text,winnings
        
            
    def isMoveValid(self,move,houseShow) :
        
        return move in self.getValidMoves(houseShow)
            
    def getCards(self) :
        
        return self.cards
        
    def isDone(self) :
        
        return self._allComplete()