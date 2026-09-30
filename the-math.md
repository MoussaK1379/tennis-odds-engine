# Tennis Prediction Model — The Math

Everything here is written to be coded directly. Order of the pipeline: ratings → serve-point probability → Markov match probability → calibration → EV/staking. Calibration is its own section at the end because it's the thing that decides whether the rest is worth anything.

---

## 1. Surface-weighted ELO (the backbone)

### Expected result
Player A's expected score against B (1 = win, 0 = loss):

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

This is also a usable match win probability on its own.

### Update after a match
$$
R_A' = R_A + K\,(S_A - E_A)
$$

where $S_A = 1$ if A won, $0$ if A lost. B updates symmetrically (its $S_B = 1 - S_A$, $E_B = 1 - E_A$).

### Dynamic K-factor
Don't use a fixed K. New players' ratings should move fast, veterans' slow. A proven form (FiveThirtyEight tennis):

$$
K_i = \frac{250}{(m_i + 5)^{0.4}}
$$

$m_i$ = number of matches player $i$ has played. Use each player's own $K$ in their own update.

### Surface weighting
Keep two ratings per player: an overall ELO and a surface-specific ELO (separate clay/grass/hard chains). The rating you actually feed the prediction is a blend:

$$
R_{\text{eff}} = w\, R_{\text{surface}} + (1 - w)\, R_{\text{overall}}
$$

Tune $w$ per surface. Clay and grass are more specialized, so they take a higher $w$ (more weight on surface history) — roughly $w \approx 0.5\text{–}0.6$ for clay/grass, lower for hard. Tune it on your own backtest, don't trust the round numbers.

Update both the overall chain and the surface chain after every match.

---

## 2. From ratings/stats to serve-point probability

The Markov match model needs one number per player: probability of winning a point **on their own serve** in this specific matchup. Estimate it by decomposing every player into serve skill and return skill (Barnett–Clarke).

Definitions across the tour (surface-specific):
- $f_t$ = tour-average fraction of points won **on serve** (≈ 0.62–0.65 on hard, lower on clay, higher on grass — compute it from your data)
- $f_A$ = player A's fraction of points won on serve
- $g_B$ = player B's fraction of points won on return
- tour-average return rate is just $1 - f_t$

A's probability of winning a point while serving against B:

$$
p_A = f_t + (f_A - f_t) - (g_B - (1 - f_t))
$$

In words: start from tour average, add A's serving edge over average, subtract B's returning edge over average. Compute $p_B$ the same way with the roles flipped (B serving, A returning).

Compute $f_A$, $g_B$ on surface-specific data, or apply a surface multiplier if you're short on samples.

---

## 3. Markov match model

Now turn $p_A$ and $p_B$ into a match win probability. It nests: point → game → set → match.

### Game (server wins their service game)
Let $p$ = server's point-win probability. To win a game you need 4 points, win by 2:

$$
P_{\text{game}}(p) = p^4 + 4p^4(1-p) + 10p^4(1-p)^2 + 20p^3(1-p)^3 \cdot \underbrace{\frac{p^2}{p^2 + (1-p)^2}}_{\text{win from deuce}}
$$

The four terms are: win to love (4–0), to 15 (4–1), to 30 (4–2), and reach deuce (3–3) then win it.

**Deuce derivation** — from deuce you either win the next two points, or split and return to deuce:

$$
P_{\text{deuce}} = p^2 + 2p(1-p)P_{\text{deuce}} \;\Rightarrow\; P_{\text{deuce}} = \frac{p^2}{p^2 + (1-p)^2}
$$

So $g_A = P_{\text{game}}(p_A)$ is A's hold probability; A's break probability on B's serve is $1 - P_{\text{game}}(p_B)$.

### Set (best to 6, win by 2, tiebreak at 6–6)
There's no clean closed form because serve alternates game to game, so use a small dynamic program. Track state (A's games, B's games, who serves):

$$
F(a, b, \text{server}) = \text{P(A wins set from this state)}
$$

Transition from each state on the current server's hold/break probability:
- if A serves: with prob $g_A$ go to $(a+1, b)$, else $(a, b+1)$
- if B serves: with prob $g_B$ go to $(a, b+1)$, else $(a+1, b)$
- server flips each game

Terminal states: A reaches 6 with a 2-game lead → return 1; B does → return 0; at $(6,6)$ → return the tiebreak probability below. Memoize; it's tiny.

### Tiebreak (first to 7, win by 2)
Same idea, separate DP. State = (A points, B points, who serves), with the 1–2–2 serve rotation (A serves point 1, then players alternate every 2 points). Win at 7 with a 2-point margin; from 6–6 it's the same deuce-style recursion as the game. Compute once per matchup from $p_A, p_B$.

Call the resulting set-win probability $s = P(\text{A wins a set})$.

### Match
Treating sets as independent (standard approximation — good enough; serve carryover and fatigue are second-order):

**Best of 3:**
$$
P_{\text{match}} = s^2(3 - 2s)
$$

**Best of 5:**
$$
P_{\text{match}} = s^3\,(6s^2 - 15s + 10)
$$

Score-line probabilities (2–0, 2–1, 3–1, etc.) fall out of the same expansion — that's the bonus of the mechanistic model:
- Bo3: $P(2\text{–}0)=s^2$, $\;P(2\text{–}1)=2s^2(1-s)$
- Bo5: $P(3\text{–}0)=s^3$, $\;P(3\text{–}1)=3s^3(1-s)$, $\;P(3\text{–}2)=6s^3(1-s)^2$

You now have two independent match probabilities — one from ELO, one from the Markov serve model. They make different errors, so ensemble them (weighted average of the probabilities, or feed both as features into a logistic/gradient-boosted model). Calibrate the **final** output, not the pieces.

---

## 4. Turning probability into a bet

### Strip the vig first
Book offers decimal odds $d_A, d_B$. Raw implied probabilities $q_A = 1/d_A$, $q_B = 1/d_B$ sum to more than 1 — the excess is the bookmaker's margin (overround = $q_A + q_B - 1$). Remove it (proportional method, the simple baseline):

$$
q_A^* = \frac{q_A}{q_A + q_B}, \qquad q_B^* = \frac{q_B}{q_A + q_B}
$$

(Shin or logarithmic de-vigging are better and worth upgrading to later, but start here.)

### Expected value
For a \$1 bet on A at odds $d_A$ given your **calibrated** model probability $p_A$:

$$
\text{EV} = p_A\, d_A - 1
$$

Bet only when $\text{EV} > 0$, i.e. $p_A > 1/d_A$. Your edge is $p_A d_A - 1$.

### Stake sizing — fractional Kelly
$$
f^\* = \frac{p\,d - 1}{d - 1} = \frac{bp - q}{b}, \quad b = d - 1,\; q = 1 - p
$$

Full Kelly is growth-optimal but assumes your probabilities are exactly right — they aren't, so it's far too volatile in practice. Use a fraction $\lambda$:

$$
\text{stake} = \lambda \, f^\* \cdot (\text{bankroll}), \qquad \lambda \in [0.25,\, 0.5]
$$

---

## 5. Calibration — the part that has to be right

The honest framing up front: you cannot make calibration *permanently* right and then forget it. A model is calibrated *on a dataset, at a point in time, for a segment*. Calibration drifts as the tour changes, players age, and surfaces play differently season to season. "Always right" in practice means **continuously measured and re-fit**, not solved once. Below is how to measure it, fix it, and keep it honest.

For betting this matters more than accuracy. A model that picks winners 70% of the time but says "90%" when it means "70%" will hand you positive-EV signals that are actually negative-EV, and Kelly will size into them hard. Calibration is what makes the EV math in Section 4 trustworthy.

### Step 1 — Measure it

**Reliability diagram.** Bin predictions (say 10 bins of width 0.1). For each bin plot mean predicted probability (x) against observed win frequency (y). Perfect calibration sits on the diagonal. Points below the line = overconfident (the usual failure).

**Expected Calibration Error** — single number summarizing the diagram:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\bigl|\,\text{acc}(b) - \text{conf}(b)\,\bigr|
$$

$n_b$ = predictions in bin $b$, $N$ = total, $\text{conf}(b)$ = mean predicted prob in the bin, $\text{acc}(b)$ = observed win frequency in the bin.

**Brier score** — overall probabilistic accuracy (lower is better):

$$
\text{BS} = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2, \qquad o_i \in \{0,1\}
$$

**Log loss** — punishes confident wrong predictions hardest, which is exactly the betting failure mode:

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\bigl[\,o_i \log p_i + (1 - o_i)\log(1 - p_i)\,\bigr]
$$

### Step 2 — Fix it

Fit a calibration map on a **held-out calibration set** — not your training data, not your final test data. Three options:

**Platt scaling** (logistic, the default; few parameters, robust on small data). Take the model's log-odds $z = \log\frac{p}{1-p}$ and fit two scalars $A, B$ by minimizing log loss:

$$
p_{\text{cal}} = \frac{1}{1 + \exp(-(A z + B))}
$$

**Isotonic regression** — fits any non-decreasing mapping raw→calibrated. More flexible, but needs more data and will overfit a small calibration set. Use only with a few thousand+ matches.

**Beta calibration** — operates on $[0,1]$ directly and handles the S-shaped miscalibration typical of these models better than Platt. Good upgrade once the basics work.

### Step 3 — Keep it right (this is the "always" part)

1. **Three-way split.** Train, calibrate, test must be separate. Calibrating on training data gives a fake-perfect diagram.
2. **Calibrate per segment, not just globally.** A model can be calibrated overall and badly miscalibrated on clay, on Bo5, on big underdogs, or on the WTA tour. Check reliability diagrams sliced by surface, tour, tournament tier, and favorite/underdog. Re-fit per segment if a slice is off.
3. **Watch the tails.** Heavy favorites (p > 0.9) and big dogs are where calibration breaks and where odd longshot value lives. Bin them finely.
4. **Re-fit on a rolling window.** Recompute the calibration map periodically (e.g., rolling 12 months) so it tracks drift. Monitor ECE over time; a rising ECE is your signal to re-fit.
5. **Calibrate the final ensemble output**, after blending ELO + Markov + any ML, not the components.

### Backtesting guards (so the numbers aren't lying to you)
- **No look-ahead.** All ratings/stats must be strictly as-of pre-match. The single most common way to fool yourself.
- **Test against the closing line, not opening odds.** The closing line is the sharpest market estimate; beating it is the real proof of edge.
- **Apply vig and realistic fills** in the backtest, not theoretical fair odds.
- **Check ROI stability**, not just total profit — a "winning" backtest carried by three longshots isn't an edge. Break ROI down by surface, tier, and time period.

---

## Build order
1. Surface-weighted ELO + dynamic K
2. Logistic regression on features (ELO diff, serve/return, form, fatigue) as a benchmark
3. Markov serve model for mechanistic probabilities + score lines
4. Ensemble the two
5. **Calibrate the output and verify per-segment** before risking a cent
6. Backtest against closing lines with vig applied
7. De-vig live odds → EV → fractional Kelly

Get steps 1, 2, and 5 working end to end before touching anything fancy. A calibrated simple model beats an uncalibrated sophisticated one every time.

---

*One honest note since this is for real-money betting: even a well-built, calibrated model gives a thin edge that the market erodes as it sharpens, and no calibration holds forever. Treat the responsible-gambling tooling (limits, loss alerts, self-exclusion) as core features, not afterthoughts.*
