card_map = {i: k for i, k in enumerate(list(range(2, 11)) + ["J", "Q", "K", "A"])}

card_values = {k: k for k in range(2, 11)}
for c in ["J", "Q", "K"]:
    card_values[c] = 10
card_values["A"] = 1

rules_common = {
    "dealerHitSoft17": False,
    "pushDealer22": False,
    "doubleAfterSplit": True,
    "hitAfterSplitAces": False,
    "reducedBlackjackPayout": False,
    "allowLateSurrender": True,
}
