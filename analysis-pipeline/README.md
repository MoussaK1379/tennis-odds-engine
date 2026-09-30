# Tennis Prediction Pipeline

ELO + Markov serve model, calibrated, with a betting layer. Runs out of the box
on synthetic data — press **Run**, then swap in real matches when you're ready.

## Run on Replit
1. Create a Python Repl and upload these files (keep them in the same folder).
2. Dependencies install from `requirements.txt` automatically. If you hit
   `ModuleNotFoundError`, open the **Shell** and run:
   `pip install -r requirements.txt`
3. Press **Run** (or `python main.py`).

You'll get test-set metrics, a raw-vs-calibrated reliability table, two saved
reliability plots (`reliability_raw.png`, `reliability_calibrated.png`), and the
betting demo.

## Files
- `elo.py` — surface-weighted ELO, dynamic K-factor
- `markov.py` — point → game → set → tiebreak → match probability (Barnett–Clarke serve split)
- `betting.py` — de-vig, expected value, fractional Kelly
- `calibration.py` — ECE / Brier / log loss, Platt + isotonic, reliability table & plot
- `data.py` — synthetic match generator + real-data (Sackmann) loader
- `main.py` — ties it all together (train → ensemble → calibrate → backtest → bet)

## Reading the output
- **logloss / brier / ECE**: lower is better. ECE is calibration error — for
  betting it matters more than raw accuracy.
- **Reliability table**: "pred" vs "obs" per probability bin. Close columns =
  well calibrated (the diagonal on the plot).
- **Betting demo**: *illustrative only.* The market is simulated as an
  independent noisy estimate of the true probability plus a 4.5% margin. Note
  the flat-stake edge can be positive while Kelly still draws down hard — that's
  the real lesson about adverse selection and tail calibration, not a bug.

## Swap in real data
Grab Jeff Sackmann's ATP match files (`atp_matches_YYYY.csv`) from
https://github.com/JeffSackmann/tennis_atp, then in `main.py` replace:

```python
df = generate_synthetic_matches()
```
with:
```python
from data import load_sackmann_csv
df = load_sackmann_csv("atp_matches_2024.csv")  # or concat several years, sorted by date
```

Two things to fix for real data:
- **Sort chronologically** before the walk (ELO must update in match order).
- The loader sets player A = winner every row, so labels are all `True`.
  Shuffle each match's A/B assignment (and flip `a_won`) so the model trains on
  balanced outcomes.

## Build order (if extending)
ELO → logistic baseline → Markov → ensemble → **calibrate & verify per-segment**
→ backtest against closing lines with vig → de-vig live odds → fractional Kelly.
A calibrated simple model beats an uncalibrated sophisticated one every time.

---
*For real-money use: a real market is sharper than this simulation, edges are
thin and decay, and no calibration holds forever — re-fit on a rolling window
and treat responsible-gambling limits as core features.*
