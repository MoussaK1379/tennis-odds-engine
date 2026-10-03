"""The Tennis Odds Engine model, in one place.

The daily updater (update_data.py), the match list (update_matches.py) and the
backtest (tools/backtest.py) all use this module, so what is backtested is what
runs. The website repeats predict_export() in JavaScript (index.html).

Two halves, blended per tour:
  - Rating: Elo or Glicko-2 on match results, with a per-surface offset on top
    of each player's overall rating (so a new surface starts from the overall
    rating, not from scratch).
  - Serve model: point-based serve and return ratings (share of points won,
    recency-weighted, adjusted for opponent strength and for how fast each
    tournament plays), fed through a point -> game -> set -> match Markov chain.

Matches are processed in date order and every prediction uses only what came
before it, so the same code gives honest backtest numbers.
"""

import math
from dataclasses import dataclass, asdict, fields
from functools import lru_cache

SURFACES = ("hard", "clay", "grass")
GLICKO_SCALE = 400 / math.log(10)            # 173.7178: Glicko-2 <-> Elo points


# ---------- Markov chain (mirrored in index.html) ------------------------------
def clip(p, lo=0.01, hi=0.99): return max(lo, min(hi, p))

def game_win_prob(p):
    q = 1 - p
    den = p * p + q * q
    pd = (p * p) / den if den > 0 else 0.5
    return p**4 + 4 * p**4 * q + 10 * p**4 * q * q + 20 * p**3 * q**3 * pd

def _tb_server_is_a(a, b): return True if a + b == 0 else ((a + b - 1) // 2) % 2 == 1

def tiebreak_win_prob(pa, pb):
    a_s, a_r = pa, 1 - pb
    den = a_s * a_r + (1 - a_s) * (1 - a_r)
    tail = (a_s * a_r) / den if den > 0 else 0.5

    @lru_cache(maxsize=None)
    def f(a, b):
        if a == b and a >= 6: return tail
        if a >= 7 and a - b >= 2: return 1.0
        if b >= 7 and b - a >= 2: return 0.0
        ppt = pa if _tb_server_is_a(a, b) else 1 - pb
        return ppt * f(a + 1, b) + (1 - ppt) * f(a, b + 1)
    return f(0, 0)

def set_win_prob(pa, pb):
    ga, gb, tb = game_win_prob(pa), game_win_prob(pb), tiebreak_win_prob(pa, pb)

    @lru_cache(maxsize=None)
    def f(a, b, a_serves):
        if a == 6 and b == 6: return tb
        if a >= 6 and a - b >= 2: return 1.0
        if b >= 6 and b - a >= 2: return 0.0
        pg = ga if a_serves else 1 - gb
        return pg * f(a + 1, b, not a_serves) + (1 - pg) * f(a, b + 1, not a_serves)
    return 0.5 * f(0, 0, True) + 0.5 * f(0, 0, False)

def match_from_set(s, best_of):
    return s * s * (3 - 2 * s) if best_of == 3 else s**3 * (6 * s * s - 15 * s + 10)


class SetTable:
    """set_win_prob on a grid with bilinear interpolation: the backtest calls it
    hundreds of thousands of times. Error vs the exact value is under 0.001."""
    LO, HI, STEP = 0.20, 0.95, 0.005

    def __init__(self):
        n = int(round((self.HI - self.LO) / self.STEP)) + 1
        self.n = n
        xs = [self.LO + i * self.STEP for i in range(n)]
        self.t = [[set_win_prob(a, b) for b in xs] for a in xs]

    def __call__(self, pa, pb):
        def idx(p):
            p = min(max(p, self.LO), self.HI - 1e-9)
            x = (p - self.LO) / self.STEP
            i = int(x)
            return i, x - i
        i, fx = idx(pa); j, fy = idx(pb)
        t = self.t
        return ((1 - fx) * (1 - fy) * t[i][j] + fx * (1 - fy) * t[i + 1][j]
                + (1 - fx) * fy * t[i][j + 1] + fx * fy * t[i + 1][j + 1])


# ---------- parameters ---------------------------------------------------------
@dataclass
class Params:
    # rating system: "elo" (surface offsets), "glicko" (Glicko-2 + surface
    # offsets) or "legacy" (the original separate per-surface Elos from 1500)
    rating: str = "elo"
    k0: float = 250.0            # Elo K = k0 / (matches + k_off) ** k_exp
    k_off: float = 5.0
    k_exp: float = 0.4
    surf_k: float = 1.0          # surface offset learning rate, as a share of K
    g_tau: float = 0.5           # Glicko-2 volatility constraint
    g_rd0: float = 350.0         # starting rating deviation (Elo points)
    g_days: float = 30.0         # days per Glicko rating period (drives RD growth when idle)
    g_surf_k: float = 20.0       # Glicko: surface offset step (Elo points per unit surprise)
    # serve / return ratings
    half_life: float = 365.0     # days; older points count half as much per half-life
    shrink: float = 400.0        # pseudo-points pulling each player toward the tour average
    surf_shrink: float = 1500.0  # pseudo-points pulling a surface rate toward the player's overall;
                                 # 0 = original rule (own surface rate after 3 matches)
    court_shrink: float = 0.0    # pseudo-points for tournament speed; 0 = no court adjustment
    # blend: weight on the serve model, per tour, scaled down for thin samples
    w_atp: float = 0.5
    w_wta: float = 0.5
    n0: float = 0.0              # serve points at which the weight reaches half its full value; 0 = flat

    def weight(self, tour, n):
        w = self.w_atp if tour == "ATP" else self.w_wta
        return w if self.n0 <= 0 else w * n / (n + self.n0)

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


# ---------- export-based prediction (what update_matches.py and the site use) ---
def predict_export(A, B, surface, best_of, tour_avg, model, court=0.0, set_prob=set_win_prob):
    """A, B: player entries from players.json. model: players.json["model"].
    Returns the pieces shown on the site. Mirrored by compute() in index.html."""
    def pick(obj, key, default):
        if not obj: return default
        v = obj.get(key)
        return obj.get("overall", default) if v is None else v
    T = tour_avg
    ra, rb = pick(A.get("elo"), surface, 1500), pick(B.get("elo"), surface, 1500)
    if model.get("rating") == "glicko":
        q = math.log(10) / 400
        rd2 = (A.get("rd") or 0) ** 2 + (B.get("rd") or 0) ** 2
        g = 1 / math.sqrt(1 + 3 * q * q * rd2 / math.pi ** 2)
    else:
        g = 1.0
    p_elo = 1 / (1 + 10 ** (-g * (ra - rb) / 400))
    sa, rta = pick(A.get("serve"), surface, T), pick(A.get("return"), surface, 1 - T)
    sb, rtb = pick(B.get("serve"), surface, T), pick(B.get("return"), surface, 1 - T)
    pa = clip(T + (sa - T) - (rtb - (1 - T)) + court)
    pb = clip(T + (sb - T) - (rta - (1 - T)) + court)
    s = set_prob(pa, pb)
    p_mkv = match_from_set(s, best_of)
    w_full = model.get("w", {}).get(A.get("tour", "ATP"), 0.5)
    n0 = model.get("n0", 0) or 0
    n = min(A.get("n", 0) or 0, B.get("n", 0) or 0)
    w = w_full if n0 <= 0 else w_full * n / (n + n0)
    return {"p": w * p_mkv + (1 - w) * p_elo, "p_elo": p_elo, "p_mkv": p_mkv,
            "w": w, "pa": pa, "pb": pb, "set": s}


# ---------- the online engine (one per tour) ------------------------------------
def _days(a, b): return (a - b).days


class TourEngine:
    def __init__(self, tour, params, set_prob=set_win_prob):
        self.tour, self.P, self.set_prob = tour, params, set_prob
        self.T_default = 0.64 if tour == "ATP" else 0.56
        self.T_num = self.T_den = 0.0
        self.T_last = None
        self.elo, self.off, self.played = {}, {}, {}
        self.gl = {}                      # player -> [mu, phi, sigma, last date]
        self.sv = {}                      # player -> {"last": date, key: [sw, sp, rw, rp, n]}
        self.court = {}                   # court id -> [sum resid*pts, sum pts]
        self.info = {}                    # player -> activity record for flags

    # --- tour average serve points won ---
    def T(self, d=None):
        if self.T_den <= 0:
            return self.T_default
        return self.T_num / self.T_den

    def _decay_T(self, d):
        if self.T_last is not None:
            f = 0.5 ** (_days(d, self.T_last) / self.P.half_life)
            self.T_num *= f; self.T_den *= f
        self.T_last = d

    # --- serve / return ratings ---
    def _sums(self, p, key, d):
        e = self.sv.get(p)
        if not e or key not in e:
            return 0.0, 0.0, 0.0, 0.0, 0
        f = 0.5 ** (_days(d, e["last"]) / self.P.half_life)
        sw, sp, rw, rp, n = e[key]
        return sw * f, sp * f, rw * f, rp * f, n

    def rates(self, p, surface, d):
        """(serve rate, return rate, effective serve points) for p on surface at date d."""
        T, P = self.T(), self.P
        sw, sp, rw, rp, _ = self._sums(p, "overall", d)
        S_all = (sw + P.shrink * T) / (sp + P.shrink)
        R_all = (rw + P.shrink * (1 - T)) / (rp + P.shrink)
        if not surface:
            return S_all, R_all, sp
        ssw, ssp, srw, srp, sn = self._sums(p, surface, d)
        if P.surf_shrink > 0:
            S = (ssw + P.surf_shrink * S_all) / (ssp + P.surf_shrink)
            R = (srw + P.surf_shrink * R_all) / (srp + P.surf_shrink)
        else:                          # original rule: own surface rate once there are 3 matches
            S = ssw / ssp if sn >= 3 and ssp > 0 else S_all
            R = srw / srp if sn >= 3 and srp > 0 else R_all
        return S, R, sp

    def _add(self, p, key, d, sw=0.0, sp=0.0, rw=0.0, rp=0.0, count=False):
        e = self.sv.setdefault(p, {"last": d})
        if e["last"] != d:
            f = 0.5 ** (_days(d, e["last"]) / self.P.half_life)
            for k, v in e.items():
                if k != "last":
                    v[0] *= f; v[1] *= f; v[2] *= f; v[3] *= f
            e["last"] = d
        v = e.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
        v[0] += sw; v[1] += sp; v[2] += rw; v[3] += rp
        if count: v[4] += 1

    def court_effect(self, cid):
        if not cid or self.P.court_shrink <= 0:
            return 0.0
        c = self.court.get(cid)
        return c[0] / (c[1] + self.P.court_shrink) if c else 0.0

    # --- ratings ---
    def _k(self, p):
        return self.P.k0 / (self.played.get(p, 0) + self.P.k_off) ** self.P.k_exp

    def _glicko(self, p, d):
        """Current [mu, phi, sigma] with phi grown for the time since p last played."""
        P = self.P
        phi0 = P.g_rd0 / GLICKO_SCALE
        g = self.gl.get(p)
        if not g:
            return [0.0, phi0, 0.06]
        mu, phi, sigma, last = g
        t = max(0, _days(d, last)) / P.g_days
        phi = min(math.sqrt(phi * phi + sigma * sigma * t), phi0)
        return [mu, phi, sigma]

    def rating(self, p, surface, d):
        """(rating in Elo points incl. surface offset, rating deviation in Elo points)."""
        mode = self.P.rating
        if mode == "legacy":
            r = self.off.get((p, surface))
            return (r if r is not None else self.elo.get(p, 1500.0)), 0.0
        off = self.off.get((p, surface), 0.0) if surface else 0.0
        if mode == "glicko":
            mu, phi, _ = self._glicko(p, d)
            return 1500.0 + GLICKO_SCALE * mu + off, GLICKO_SCALE * phi
        return self.elo.get(p, 1500.0) + off, 0.0

    # --- prediction ---
    def predict(self, a, b, surface, best_of, d, court=None):
        P = self.P
        T = self.T()
        ra, da = self.rating(a, surface, d)
        rb, db = self.rating(b, surface, d)
        if P.rating == "glicko":
            q = math.log(10) / 400
            g = 1 / math.sqrt(1 + 3 * q * q * (da * da + db * db) / math.pi ** 2)
        else:
            g = 1.0
        p_elo = 1 / (1 + 10 ** (-g * (ra - rb) / 400))
        Sa, Ra, na = self.rates(a, surface, d)
        Sb, Rb, nb = self.rates(b, surface, d)
        ce = self.court_effect(court)
        pa = clip(T + (Sa - T) - (Rb - (1 - T)) + ce)
        pb = clip(T + (Sb - T) - (Ra - (1 - T)) + ce)
        p_mkv = match_from_set(self.set_prob(pa, pb), best_of)
        w = P.weight(self.tour, min(na, nb))
        return {"p": w * p_mkv + (1 - w) * p_elo, "p_elo": p_elo, "p_mkv": p_mkv,
                "w": w, "n": min(na, nb)}

    # --- update with a finished match ---
    def update(self, m):
        """m: dict with date, surface, winner, loser, status ('completed'|'ret'|'wo'),
        w_srv / l_srv ((won, played) or None), court (id or None)."""
        d, s, w, l = m["date"], m.get("surface"), m["winner"], m["loser"]
        for p, won in ((w, True), (l, False)):
            i = self.info.setdefault(p, {"dates": [], "rets": [], "surf": {}})
            i["dates"].append(d)
            if s:
                i["surf"].setdefault(s, []).append(d)
        if m["status"] == "ret":
            self.info[l]["rets"].append(d)          # the player who didn't finish
        if m["status"] == "wo":
            return
        if m["status"] == "completed":
            self._update_rating(w, l, s, d)
        self._update_serve(m, d, s)

    def _update_rating(self, w, l, s, d):
        P = self.P
        if P.rating == "legacy":
            rw, rl = self.elo.get(w, 1500.0), self.elo.get(l, 1500.0)
            kw, kl = self._k(w), self._k(l)
            e = 1 / (1 + 10 ** ((rl - rw) / 400))
            self.elo[w] = rw + kw * (1 - e); self.elo[l] = rl - kl * (1 - e)
            if s:
                sw, sl = self.off.get((w, s), 1500.0), self.off.get((l, s), 1500.0)
                es = 1 / (1 + 10 ** ((sl - sw) / 400))
                self.off[(w, s)] = sw + kw * (1 - es); self.off[(l, s)] = sl - kl * (1 - es)
        elif P.rating == "elo":
            ow = self.off.get((w, s), 0.0) if s else 0.0
            ol = self.off.get((l, s), 0.0) if s else 0.0
            rw, rl = self.elo.get(w, 1500.0), self.elo.get(l, 1500.0)
            e = 1 / (1 + 10 ** (((rl + ol) - (rw + ow)) / 400))
            kw, kl = self._k(w), self._k(l)
            self.elo[w] = rw + kw * (1 - e); self.elo[l] = rl - kl * (1 - e)
            if s:
                self.off[(w, s)] = ow + P.surf_k * kw * (1 - e)
                self.off[(l, s)] = ol - P.surf_k * kl * (1 - e)
        else:
            gw, gl_ = self._glicko(w, d), self._glicko(l, d)
            ow = self.off.get((w, s), 0.0) if s else 0.0
            ol = self.off.get((l, s), 0.0) if s else 0.0
            nw = self._glicko_step(gw, gl_, 1.0, (ow - ol) / GLICKO_SCALE)
            nl = self._glicko_step(gl_, gw, 0.0, (ol - ow) / GLICKO_SCALE)
            ew = 1 / (1 + math.exp(-(gw[0] - gl_[0] + (ow - ol) / GLICKO_SCALE)))
            self.gl[w] = nw + [d]; self.gl[l] = nl + [d]
            if s:
                self.off[(w, s)] = ow + P.g_surf_k * (1 - ew)
                self.off[(l, s)] = ol - P.g_surf_k * (1 - ew)
        self.played[w] = self.played.get(w, 0) + 1
        self.played[l] = self.played.get(l, 0) + 1

    def _glicko_step(self, me, opp, score, edge):
        """One Glicko-2 rating period with a single game (Glickman 2013, steps 3-8).
        edge: surface-offset difference (in Glicko units) added to my rating."""
        mu, phi, sigma = me
        mu_j, phi_j, _ = opp
        g = 1 / math.sqrt(1 + 3 * phi_j * phi_j / math.pi ** 2)
        E = 1 / (1 + math.exp(-g * (mu + edge - mu_j)))
        v = 1 / (g * g * E * (1 - E))
        delta = v * g * (score - E)
        tau = self.P.g_tau
        a = math.log(sigma * sigma)

        def f(x):
            ex = math.exp(x)
            return (ex * (delta * delta - phi * phi - v - ex) / (2 * (phi * phi + v + ex) ** 2)
                    - (x - a) / (tau * tau))
        A = a
        if delta * delta > phi * phi + v:
            B = math.log(delta * delta - phi * phi - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
        fA, fB = f(A), f(B)
        for _ in range(60):
            if abs(B - A) <= 1e-6:
                break
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A, fA = B, fB
            else:
                fA /= 2
            B, fB = C, fC
        sigma_new = math.exp(A / 2)
        phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
        phi_new = 1 / math.sqrt(1 / (phi_star * phi_star) + 1 / v)
        mu_new = mu + phi_new * phi_new * g * (score - E)
        return [mu_new, phi_new, sigma_new]

    def _update_serve(self, m, d, s):
        P, T = self.P, self.T()
        w, l, cid = m["winner"], m["loser"], m.get("court")
        ce = self.court_effect(cid)
        # pre-match ratings for the opponent adjustment
        Sw, Rw, _ = self.rates(w, s, d)
        Sl, Rl, _ = self.rates(l, s, d)
        obs = []
        if m.get("w_srv"): obs.append((w, l, Sw, Rl) + tuple(m["w_srv"]))
        if m.get("l_srv"): obs.append((l, w, Sl, Rw) + tuple(m["l_srv"]))
        if not obs:
            return
        self._decay_T(d)
        for srv, ret, S_srv, R_ret, won, pts in obs:
            if not pts:
                continue
            rate = won / pts
            adj_s = rate - ce + (R_ret - (1 - T))       # tough returner -> serve rate counts for more
            adj_r = (1 - rate) + ce + (S_srv - T)        # big server -> return rate counts for more
            for key in ("overall", s) if s else ("overall",):
                self._add(srv, key, d, sw=adj_s * pts, sp=pts, count=True)
                self._add(ret, key, d, rw=adj_r * pts, rp=pts)
            if cid:
                expected = T + (S_srv - T) - (R_ret - (1 - T))
                c = self.court.setdefault(cid, [0.0, 0.0])
                c[0] += (rate - expected) * pts; c[1] += pts
            self.T_num += won; self.T_den += pts

    # --- export for players.json ---
    def export(self, p, d):
        e = {"elo": {}, "serve": {}, "return": {}}
        r, rd = self.rating(p, None, d)
        e["elo"]["overall"] = round(r)
        S, R, n = self.rates(p, None, d)
        e["serve"]["overall"], e["return"]["overall"] = round(S, 4), round(R, 4)
        for s in SURFACES:
            rs, _ = self.rating(p, s, d)
            e["elo"][s] = round(rs)
            Ss, Rs, _ = self.rates(p, s, d)
            e["serve"][s], e["return"][s] = round(Ss, 4), round(Rs, 4)
        if self.P.rating == "glicko":
            e["rd"] = round(rd)
        e["n"] = round(n)
        e["info"] = self.activity(p, d)
        return e

    def activity(self, p, d):
        """What the site's warnings need: recent match counts, layoff, retirements."""
        i = self.info.get(p) or {"dates": [], "rets": [], "surf": {}}
        last = max(i["dates"]) if i["dates"] else None
        return {"last": last.isoformat() if last else None,
                "m12": sum(1 for x in i["dates"] if _days(d, x) <= 365),
                "surf12": {s: sum(1 for x in v if _days(d, x) <= 365) for s, v in i["surf"].items()},
                "ret180": sum(1 for x in i["rets"] if _days(d, x) <= 180),
                "last_ret": max(i["rets"]).isoformat() if i["rets"] else None}

    def courts(self):
        return {c: round(self.court_effect(c), 4) for c in self.court} if self.P.court_shrink > 0 else {}

    def model_block(self):
        return {"rating": "glicko" if self.P.rating == "glicko" else "elo",
                "w": {"ATP": self.P.w_atp, "WTA": self.P.w_wta}, "n0": self.P.n0}
