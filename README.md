# vegachimp
Option EV calculator for the poors

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

Bla bla bla, use to learn at least change it if you're gonna be a cunt and resell.

🧠 Vol Breakout EV — Monkey Mode Guide
🪄 What This Thing Actually Does
This tool shows if buying volatility (calls, puts, or straddles) makes sense based on what you think will happen — not what the market’s pricing.
It compares:
•	what the market implies (implied vol σ_imp)
•	what you expect to happen (realized vol σ_real or a post-event scenario)
and tells you whether the trade has positive or negative EV (expected value).
In short:
💰 If σ_real > σ_imp → long options make money (on average).
💀 If σ_real ≤ σ_imp → you’re donating to the market makers.
________________________________________
⚙️ Basic Inputs
Spot (S)
Current price of the underlying.
If you’re trading SPY at $500, enter 500.
Expiry (days)
Days until the option expires.
You can use any number (e.g., 14, 21, 45, 90).
Structure
Choose one:
•	Call – single long call
•	Put – single long put
•	ATM Straddle – buy both an at-the-money call and put
Strike (K)
Option strike price. Usually same as spot for ATM.
Implied Vol (σ_imp)
The current market volatility you pay for when buying the option.
You can pull this from your broker’s option chain (or eyeball from barchart, etc).
Expected Realized Vol (σ_real)
How much you think the stock will actually move.
Example: If you expect a big post-earnings move, set this higher than σ_imp.
________________________________________
💵 Optional: Manual Mids
If you can see the real mid price of the option from your broker,
you can override the Black–Scholes model and input it manually here.
That gives you more accurate EV since it uses your real entry.
________________________________________
⚡ Quick Scenario Mode
Expander called: “⚡ Quick scenario (move + IV change, ignores probabilities)”
This is the fastest way to sanity-check a trade:
1.	Choose direction (Up/Down)
2.	Enter a move (%) (e.g., +5%)
3.	Set how much IV you think will change after the move (in vol points, not %)
4.	Choose how many days you’ll hold (Hold Days)
It instantly tells you:
•	the fair value after that move
•	the EV (fair – entry price)
This is your “What happens if it pops 5% and IV goes up 10?” playground.
________________________________________
Event / Unwind Mode
For earnings or catalysts (where IV crush happens).
Check “Event/Unwind mode (crush & gap)” to simulate it.
Settings:
•	Unwind after (days) – how long you hold before selling
•	Gap up / down (%) – size of expected move after event
•	Prob up (%) – probability of the move being up
•	IV crush call/put (%) – how much vol collapses after event
The app averages the up/down outcomes weighted by probability → gives you a fair EV under your scenario.
You’ll see:
•	Premium paid (entry)
•	Fair under scenario (expected exit value)
•	EV (expected) and Expected ROI
If your EV is negative, you’re paying too much for the move.
________________________________________
Guardrails
The app yells at you when you’re doing something dumb:
•	σ_real ≤ σ_imp → long-vol EV ≤ 0 (buying overpriced options)
•	EV ≤ 0 → “Are you planning to lose money?”
•	σ_real ≥ 2×σ_imp → probably unrealistic; check if you’re overestimating movement
•	Tiny premium → high ROI %s may be misleading
________________________________________
Breakeven Calculator
At the bottom:
•	Profit breakeven = how far price needs to move (in %) before EV = 0
(only in the logical direction: up for calls, down for puts)
•	Optional loss breakeven (inside expander) = how far it can move against you before EV flips negative.
Example:
Call profit breakeven: Up ≈ 8.2%
means the stock needs to rise 8.2% before your call breaks even.
________________________________________
How to Actually Use This
1.	Plug in current market numbers
o	Spot, strike, implied vol, expiry.
2.	Enter your expectations
o	Expected realized vol, or post-event move + IV crush.
3.	(Optional) Enter your real option mids from your broker.
4.	Check EV
o	If EV > 0, you’re underpaying for volatility (good trade, maybe).
o	If EV < 0, you’re overpaying (bad trade, probably).
5.	Use the Quick Scenario
o	to visualize how much a move + vol change affects your position.
________________________________________
TL;DR Monkey Logic
Situation	Likely Outcome
σ_real ≫ σ_imp	Long options win
σ_real ≈ σ_imp	You lose to theta
σ_real < σ_imp	You’re getting scammed
IV crush after event	You get smoked unless price gap > crush effect
Positive EV	The trade is mathematically favorable
Negative EV	You’re paying too much premium
Profit breakeven far away	Market expects fireworks; you need a miracle

FAQ: Yes, I had AI write this cause I was lazy.
