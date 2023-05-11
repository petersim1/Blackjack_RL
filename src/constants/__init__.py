const_cardMap = {i:k for i,k in enumerate(list(range(2,11)) + ['J','Q','K','A'])}

const_values = {k:k for k in range(2,11)}
for c in ['J','Q','K'] :
    const_values[c] = 10
const_values['A'] = 1

const_rules_common = {
    "dealerHitSoft17": False,
    "pushDealer22": False,
    "doubleAfterSplit": True,
    "hitAfterSplitAces": False,
    "reducedBlackjackPayout": False,
    "allowLateSurrender": True,
}