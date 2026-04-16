from __future__ import annotations
import os, json, math, itertools, textwrap
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans CN",
    "Arial Unicode MS", "PingFang SC", "Heiti SC", "WenQuanYi Zen Hei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FINAL = os.path.join(ROOT, "data", "final")
OUT        = os.path.join(ROOT, "outputs")
CFG        = os.path.join(ROOT, "config", "config.json")
# Fallback for running this script outside the project tree (e.g., in a sandbox)
if not os.path.isdir(DATA_FINAL):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_FINAL = os.path.join(ROOT, "data", "final")
    OUT = os.path.join(ROOT, "outputs")
    CFG = os.path.join(ROOT, "config", "config.json")
os.makedirs(OUT, exist_ok=True)

# Scenario levels (as in Module B)
LEVELS = {"P":[1,2,3], "I":[1,2], "G":[1,2], "S":[1,2,3], "L":[1,2,3], "C":[1,2]}
CONDS  = ["S_high","L_long","C_relief","I_high","G_help","P_high"]

DIR_EXPECT_HIGH = {"S_high": 1, "L_long": 1, "C_relief": 1, "P_high": 0}
DIR_EXPECT_LOW  = {"S_high": 0, "L_long": 0, "C_relief": 0, "P_high": 1}

DEFAULT_CONS_THR = 0.80
DEFAULT_FREQ_THR = 2.0  

def load_config() -> dict:
    if os.path.exists(CFG):
        with open(CFG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1/(1+np.exp(-z))

def calibrate_direct(y: np.ndarray, full_out: float, crossover: float, full_in: float) -> np.ndarray:
    """
    Direct calibration via a piecewise logit mapping that uses all three anchors:
      - full_out   -> 0.05 membership
      - crossover  -> 0.50 membership
      - full_in    -> 0.95 membership

    This keeps the mapping monotone while ensuring that `full_out` genuinely affects
    calibration and can therefore be meaningfully perturbed in sensitivity analysis.
    """
    y = np.asarray(y, dtype=float)
    eps = 1e-9

    if not (full_out < crossover < full_in):
        full_out = min(full_out, crossover - 1e-6)
        full_in = max(full_in, crossover + 1e-6)

    logit95 = math.log(0.95 / 0.05)
    left_scale = logit95 / max(crossover - full_out, eps)
    right_scale = logit95 / max(full_in - crossover, eps)

    z = np.empty_like(y, dtype=float)
    left_mask = y < crossover
    z[left_mask] = left_scale * (y[left_mask] - crossover)
    z[~left_mask] = right_scale * (y[~left_mask] - crossover)

    return sigmoid(z)

def fuzzy_memberships_from_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["S_high"] = df["S"].map({1:0.0, 2:0.5, 3:1.0})
    df["L_long"] = df["L"].map({1:0.0, 2:0.5, 3:1.0})
    df["C_relief"] = (df["C"]==2).astype(float)
    df["I_high"] = (df["I"]==2).astype(float)
    df["G_help"] = (df["G"]==2).astype(float)
    df["P_high"] = df["P"].map({1:0.0, 2:0.5, 3:1.0})
    return df

def outcome_with_prediction_fill(df_cases: pd.DataFrame, y_raw_col: str, pred_col: str = "mean_rating_pred") -> np.ndarray:
    """
    Apply the project rule of "observed first, prediction used only to fill missing cases".
    For already-complete outcomes (e.g., mean_rating_pred), this simply returns the raw series.
    """
    y = df_cases[y_raw_col].copy()
    if y.isna().any() and pred_col in df_cases.columns:
        y = y.fillna(df_cases[pred_col])
    return y.values.astype(float)

def build_full_scenarios() -> pd.DataFrame:
    rows=[]
    for P in LEVELS["P"]:
        for I in LEVELS["I"]:
            for G in LEVELS["G"]:
                for S in LEVELS["S"]:
                    for L in LEVELS["L"]:
                        for C in LEVELS["C"]:
                            rows.append({"P":P,"I":I,"G":G,"S":S,"L":L,"C":C})
    df = pd.DataFrame(rows)
    df["scenario_id"] = df.apply(lambda r: f"{int(r.P)}{int(r.I)}{int(r.G)}{int(r.S)}{int(r.L)}{int(r.C)}", axis=1)
    return df

# ---------- FE prediction for all 216 ----------
def _dummies_fixed(x: pd.Series, levels: List[int], prefix: str) -> pd.DataFrame:
    cat = pd.Categorical(x.astype(int).values, categories=levels, ordered=True)
    return pd.get_dummies(cat, prefix=prefix, drop_first=True, dtype=float)

def fit_within_fe(clean_fse_long_path: str):
    """
    Fit within (individual fixed effects) for vignette ratings:
      rating ~ P+I+G+S+L+C + (S×L + S×C + L×C) + (S×L×C)
    Using de-meaning by rid.
    """
    df = pd.read_csv(clean_fse_long_path)
    if "Cdim" in df.columns and "C" not in df.columns:
        df = df.rename(columns={"Cdim":"C"})
    df = df.dropna(subset=["rid","rating","P","I","G","S","L","C"]).copy()
    for c in ["P","I","G","S","L","C"]:
        df[c] = df[c].astype(int)

    y = df["rating"].values.astype(float)
    rid = pd.Categorical(df["rid"]).codes

    D={}
    for v in ["P","I","G","S","L","C"]:
        D[v] = _dummies_fixed(df[v], LEVELS[v], v)

    X = pd.concat([D[v] for v in ["P","I","G","S","L","C"]], axis=1)

    def add_interactions(a,b):
        for ca in D[a].columns:
            for cb in D[b].columns:
                X[f"{ca}:{cb}"] = D[a][ca].values * D[b][cb].values
    add_interactions("S","L"); add_interactions("S","C"); add_interactions("L","C")

    for ca in D["S"].columns:
        for cb in D["L"].columns:
            for cc in D["C"].columns:
                X[f"{ca}:{cb}:{cc}"] = D["S"][ca].values * D["L"][cb].values * D["C"][cc].values

    X = X.astype(float)
    X_np = X.values
    y_bar = pd.Series(y).groupby(rid).transform("mean").values
    X_bar = pd.DataFrame(X_np).groupby(rid).transform("mean").values
    y_dm = y - y_bar
    X_dm = X_np - X_bar
    beta, *_ = np.linalg.lstsq(X_dm, y_dm, rcond=None)

    # recover average intercept (mean alpha_i)
    ybar_g = pd.Series(y).groupby(rid).mean().values
    Xbar_g = pd.DataFrame(X_np).groupby(rid).mean().values
    alpha_g = ybar_g - Xbar_g @ beta
    alpha_bar = float(np.mean(alpha_g))

    design_cols = list(X.columns)

    def design_vec(P:int,I:int,G:int,S:int,L:int,C:int) -> np.ndarray:
        row = pd.DataFrame([{"P":P,"I":I,"G":G,"S":S,"L":L,"C":C}])
        dD={}
        for v in ["P","I","G","S","L","C"]:
            dD[v] = _dummies_fixed(row[v], LEVELS[v], v)
        Xs = pd.concat([dD[v] for v in ["P","I","G","S","L","C"]], axis=1)
        for a,b in [("S","L"),("S","C"),("L","C")]:
            for ca in dD[a].columns:
                for cb in dD[b].columns:
                    Xs[f"{ca}:{cb}"] = dD[a][ca].values * dD[b][cb].values
        for ca in dD["S"].columns:
            for cb in dD["L"].columns:
                for cc in dD["C"].columns:
                    Xs[f"{ca}:{cb}:{cc}"] = dD["S"][ca].values * dD["L"][cb].values * dD["C"][cc].values
        for col in design_cols:
            if col not in Xs.columns:
                Xs[col]=0.0
        Xs = Xs[design_cols].astype(float)
        return Xs.values.reshape(-1)

    return beta, alpha_bar, design_vec

# ---------- Necessary conditions ----------
def necessary_conditions(df: pd.DataFrame, y_col: str, conds: List[str], w: np.ndarray) -> pd.DataFrame:
    Y = df[y_col].values.astype(float)
    w = w.astype(float)
    sumY = float(np.sum(w*Y)) + 1e-12
    rows=[]
    for c in conds:
        X = df[c].values.astype(float)
        num = float(np.sum(w*np.minimum(X, Y)))
        cons = num / sumY
        cov  = num / (float(np.sum(w*X)) + 1e-12)
        rows.append({"condition": c, "type":"X", "consistency":cons, "coverage":cov})
        Xn = 1 - X
        num2 = float(np.sum(w*np.minimum(Xn, Y)))
        cons2 = num2 / sumY
        cov2  = num2 / (float(np.sum(w*Xn)) + 1e-12)
        rows.append({"condition": c, "type":"~X", "consistency":cons2, "coverage":cov2})
    return pd.DataFrame(rows).sort_values(["consistency","coverage"], ascending=False)

def assign_config_bits(df: pd.DataFrame, conds: List[str]) -> pd.Series:
    bits=[]
    for c in conds:
        bits.append((df[c].values.astype(float) >= 0.5).astype(int))
    mat = np.vstack(bits).T
    return pd.Series(["".join(map(str,row)) for row in mat])

def conj_membership_for_row(df_row: pd.DataFrame, config: str, conds: List[str]) -> np.ndarray:
    m = np.ones(len(df_row), dtype=float)
    for bit,c in zip(config, conds):
        x = df_row[c].values.astype(float)
        m = np.minimum(m, x if bit=="1" else (1-x))
    return m

def truth_table_fs(df: pd.DataFrame, conds: List[str], y: np.ndarray, w: np.ndarray,
                   cons_thr: float, freq_thr: float, pri_thr: float = 0.0) -> pd.DataFrame:
    y = y.astype(float); w = w.astype(float)
    df = df.copy()
    df["_cfg_bits"] = assign_config_bits(df, conds)

    all_cfgs = ["".join(map(str,b)) for b in itertools.product([0,1], repeat=len(conds))]
    rows=[]
    sumY = float(np.sum(w*y)) + 1e-12

    for cfg in all_cfgs:
        sub = df[df["_cfg_bits"]==cfg]
        if len(sub)==0:
            freq = 0.0
            cons = np.nan
            cov  = 0.0
            pri  = np.nan
        else:
            ww = w[sub.index.values]
            yy = y[sub.index.values]
            m = conj_membership_for_row(sub, cfg, conds)

            freq = float(np.sum(ww))  # weighted case frequency
            denom = float(np.sum(ww*m)) + 1e-12
            min_my = np.minimum(m, yy)
            cons = float(np.sum(ww*min_my)/denom)
            cov  = float(np.sum(ww*min_my)/sumY)

            # PRI
            y0 = 1-yy
            min_myc = np.minimum(m, y0)
            denom2 = float(np.sum(ww*min_my)) + 1e-12
            pri = float((np.sum(ww*min_my) - np.sum(ww*min_myc))/denom2)

        rows.append({
            "config_bits": cfg,
            "config": f"cfg_{cfg}",
            "freq": freq,
            "consistency": cons,
            "PRI": pri,
            "raw_coverage": cov
        })

    tt = pd.DataFrame(rows)
    tt["remainder"] = (tt["freq"] < freq_thr)

    if pri_thr > 0:
        tt["keep"] = (tt["consistency"] >= cons_thr) & (tt["PRI"] >= pri_thr) & (tt["freq"] >= freq_thr)
    else:
        tt["keep"] = (tt["consistency"] >= cons_thr) & (tt["freq"] >= freq_thr)

    return tt.sort_values(["keep","consistency","freq"], ascending=[False,False,False]).reset_index(drop=True)

def qm_prime_implicants(terms: List[str]) -> List[str]:
    from collections import defaultdict
    if not terms:
        return []
    groups = defaultdict(set)
    for t in terms:
        groups[t.count("1")].add(t)

    def combine(a,b):
        diff=0; out=[]
        for x,y in zip(a,b):
            if x==y: out.append(x)
            elif x!="-" and y!="-":
                diff += 1; out.append("-")
            else:
                return None
        return "".join(out) if diff==1 else None

    primes=set()
    current={k:set(v) for k,v in groups.items()}
    while True:
        next_groups=defaultdict(set)
        used=set()
        keys=sorted(current.keys())
        for i in keys:
            for a in current.get(i,set()):
                for b in current.get(i+1,set()):
                    c = combine(a,b)
                    if c is not None:
                        used.add(a); used.add(b)
                        next_groups[c.count("1")].add(c)
        for i in keys:
            for a in current.get(i,set()):
                if a not in used:
                    primes.add(a)
        if not next_groups:
            break
        current={k:set(v) for k,v in next_groups.items()}
    return sorted(primes)

def covers(imp: str, m: str) -> bool:
    for x,y in zip(imp,m):
        if x=="-": continue
        if x!=y: return False
    return True

def qm_minimize(minterms: List[str], dontcares: List[str]) -> List[str]:
    terms = sorted(set(minterms + dontcares))
    primes = qm_prime_implicants(terms)

    uncovered=set(minterms)
    chosen=[]
    while uncovered:
        best=None; best_cov=0
        for imp in primes:
            cov_n = sum(1 for m in uncovered if covers(imp,m))
            if cov_n > best_cov:
                best_cov = cov_n; best = imp
        if best is None:
            break
        chosen.append(best)
        uncovered = set(m for m in uncovered if not covers(best,m))
    return chosen

def implicant_membership(df: pd.DataFrame, imp: str, conds: List[str]) -> np.ndarray:
    m = np.ones(len(df), dtype=float)
    for bit,c in zip(imp,conds):
        if bit=="-": continue
        x = df[c].values.astype(float)
        m = np.minimum(m, x if bit=="1" else (1-x))
    return m

def implicant_metrics(df: pd.DataFrame, imp: str, conds: List[str], y: np.ndarray, w: np.ndarray) -> Dict[str,float]:
    m = implicant_membership(df, imp, conds)
    denom = float(np.sum(w*m)) + 1e-12
    min_my = np.minimum(m, y)
    cons = float(np.sum(w*min_my)/denom)
    cov  = float(np.sum(w*min_my)/(float(np.sum(w*y))+1e-12))
    freq = float(np.sum(w*m))
    return {"freq":freq, "consistency":cons, "raw_coverage":cov}

def solution_overall_metrics(df: pd.DataFrame, implicants: List[str], conds: List[str], y: np.ndarray, w: np.ndarray) -> Dict[str,float]:
    if not implicants:
        return {"solution_consistency": np.nan, "solution_coverage": 0.0}
    ms = [implicant_membership(df, imp, conds) for imp in implicants]
    m_sol = np.maximum.reduce(ms)
    denom = float(np.sum(w*m_sol)) + 1e-12
    min_my = np.minimum(m_sol, y)
    sol_cons = float(np.sum(w*min_my)/denom)
    sol_cov  = float(np.sum(w*min_my)/(float(np.sum(w*y))+1e-12))
    return {"solution_consistency": sol_cons, "solution_coverage": sol_cov}

def implicants_to_expression(imps: List[str], conds: List[str]) -> List[str]:
    out=[]
    for imp in imps:
        terms=[]
        for bit,c in zip(imp,conds):
            if bit=="-": continue
            terms.append(c if bit=="1" else f"~{c}")
        out.append(" * ".join(terms) if terms else "1")
    return out

def directional_compatible(cfg: str, conds: List[str], expect: Dict[str,int]) -> bool:
    for bit,c in zip(cfg,conds):
        if c not in expect: 
            continue
        if int(bit) != int(expect[c]):
            return False
    return True

def solve(tt: pd.DataFrame, df_cases: pd.DataFrame, conds: List[str], y: np.ndarray, w: np.ndarray,
          cons_thr: float, freq_thr: float, pri_thr: float, expect: Dict[str,int], label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    minterms = tt[tt["keep"]]["config_bits"].tolist()
    remainders = tt[tt["remainder"]]["config_bits"].tolist()
    remainders_dir = [c for c in remainders if directional_compatible(c, conds, expect)]

    results=[]
    overall=[]
    for sol_type, dontcares in [
        ("complex", []),
        ("intermediate", remainders_dir),
        ("parsimonious", remainders),
    ]:
        imps = qm_minimize(minterms=minterms, dontcares=dontcares)
        exprs = implicants_to_expression(imps, conds)
        for imp, expr in zip(imps, exprs):
            met = implicant_metrics(df_cases, imp, conds, y, w)
            results.append({"label":label, "solution_type":sol_type, "implicant":imp, "expression":expr, **met})
        om = solution_overall_metrics(df_cases, imps, conds, y, w)
        overall.append({"label":label, "solution_type":sol_type, "n_minterms":len(minterms), "n_remainders":len(remainders),
                        "n_dontcares_used":len(dontcares), **om})
        if not imps:
            results.append({"label":label, "solution_type":sol_type, "implicant":"", "expression":"(no solution)", 
                            "freq":0.0, "consistency":np.nan, "raw_coverage":0.0})
    return pd.DataFrame(results), pd.DataFrame(overall)

def sensitivity_grid(df_cases: pd.DataFrame, y_raw_col: str, w: np.ndarray,
                     full_in_list: List[float], cross_list: List[float], full_out_list: List[float],
                     cons_thr_list: List[float], freq_thr_list: List[float], pri_thr_list: List[float]) -> pd.DataFrame:
    rows=[]
    y_base = outcome_with_prediction_fill(df_cases, y_raw_col)
    for full_out in full_out_list:
        for cross in cross_list:
            for full_in in full_in_list:
                Y = calibrate_direct(y_base, full_out, cross, full_in)
                for cons_thr in cons_thr_list:
                    for freq_thr in freq_thr_list:
                        for pri_thr in pri_thr_list:
                            tt = truth_table_fs(df_cases, CONDS, y=Y, w=w, cons_thr=cons_thr, freq_thr=freq_thr, pri_thr=pri_thr)
                            # keep count and coverage of kept rows
                            kept = tt[tt["keep"]]
                            rows.append({
                                "y_source": y_raw_col,
                                "full_out": full_out,
                                "crossover": cross,
                                "full_in": full_in,
                                "cons_thr": cons_thr,
                                "freq_thr": freq_thr,
                                "pri_thr": pri_thr,
                                "n_kept": int(len(kept)),
                                "kept_cov_sum": float(np.nansum(kept["raw_coverage"].values)) if len(kept) else 0.0,
                                "kept_freq_sum": float(np.nansum(kept["freq"].values)) if len(kept) else 0.0,
                                "max_consistency": float(np.nanmax(tt["consistency"].values)),
                                "min_consistency": float(np.nanmin(tt["consistency"].values)),
                            })
    return pd.DataFrame(rows)

def parse_recipe_expression(expr: str) -> List[Tuple[str, bool]]:
    """
    expr like: "S_high * L_long * ~P_high"
    return list of (cond, is_positive)
    """
    expr = expr.strip()
    if expr in ("", "(no solution)", "1"):
        return []
    parts = [p.strip() for p in expr.split("*")]
    lits=[]
    for p in parts:
        if p.startswith("~"):
            lits.append((p[1:].strip(), False))
        else:
            lits.append((p.strip(), True))
    return lits

def cases_satisfying_recipe(df_cases: pd.DataFrame, recipe_expr: str, threshold: float = 0.5) -> pd.DataFrame:
    lits = parse_recipe_expression(recipe_expr)
    if not lits:
        return df_cases.copy()
    mask = np.ones(len(df_cases), dtype=bool)
    for cond, pos in lits:
        if cond not in df_cases.columns:
            continue
        x = df_cases[cond].values.astype(float)
        if pos:
            mask &= (x >= threshold)
        else:
            mask &= (x < threshold)
    return df_cases.loc[mask].copy()

def summarize_recipe_mapping(df_cases: pd.DataFrame, recipe_rows: pd.DataFrame, b3_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    out_rows=[]
    for _,r in recipe_rows.iterrows():
        if r.get("solution_type") != "intermediate":
            continue
        expr = str(r.get("expression","")).strip()
        if expr in ("", "(no solution)"):
            continue
        sub = cases_satisfying_recipe(df_cases, expr, threshold=0.5)
        contexts = sorted({f"{int(a)}{int(b)}{int(c)}" for a,b,c in zip(sub["P"], sub["I"], sub["G"])})
        packages = sorted({f"S{int(s)}-L{int(l)}-C{int(c)}" for s,l,c in zip(sub["S"], sub["L"], sub["C"])})
        row = {
            "label": r.get("label"),
            "solution_type": r.get("solution_type"),
            "expression": expr,
            "implicant": r.get("implicant"),
            "recipe_n_cases": int(len(sub)),
            "contexts_n": int(len(contexts)),
            "contexts_list": ";".join(contexts),
            "packages_n": int(len(packages)),
            "packages_list": ";".join(packages),
            "recipe_freq": float(r.get("freq", np.nan)),
            "recipe_consistency": float(r.get("consistency", np.nan)),
            "recipe_coverage": float(r.get("raw_coverage", np.nan)),
        }
        # If Module-B table exists, compute best packages within this recipe by avg mu_hat
        if b3_table is not None and len(sub)>0:
            # Build keys
            sub_keys = sub[["P","I","G","S","L","C"]].copy()
            sub_keys["Cdim"] = sub_keys["C"]
            sub_keys["label_pkg"] = sub_keys.apply(lambda x: f"S{int(x.S)}-L{int(x.L)}-C{int(x.Cdim)}", axis=1)
            sub_keys["ctx"] = sub_keys.apply(lambda x: f"{int(x.P)}{int(x.I)}{int(x.G)}", axis=1)
            # Join with b3_table (expected columns)
            bt = b3_table.copy()
            # tolerate naming
            if "Cdim" not in bt.columns and "C" in bt.columns:
                bt["Cdim"]=bt["C"]
            # define join columns
            join_cols = []
            for col in ["P","I","G","S","L","Cdim"]:
                if col in bt.columns:
                    join_cols.append(col)
            if set(["P","I","G","S","L"]).issubset(set(bt.columns)) and "Cdim" in bt.columns:
                merged = pd.merge(sub_keys, bt, on=["P","I","G","S","L","Cdim"], how="left")
                if "mu_hat" in merged.columns:
                    pkg_score = merged.groupby("label_pkg")["mu_hat"].mean().sort_values(ascending=False)
                    top = pkg_score.head(5)
                    row["top5_packages_by_mu_hat"] = ";".join([f"{k}:{v:.3f}" for k,v in top.items()])
                    row["top5_mu_hat_mean"] = float(top.mean()) if len(top) else np.nan
                if "delta_vs_baseline" in merged.columns:
                    pkg_delta = merged.groupby("label_pkg")["delta_vs_baseline"].mean().sort_values(ascending=False)
                    topd = pkg_delta.head(5)
                    row["top5_packages_by_delta"] = ";".join([f"{k}:{v:.3f}" for k,v in topd.items()])
                    row["top5_delta_mean"] = float(topd.mean()) if len(topd) else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# ---------- Path-card plotting ----------
CARD_STYLE = {
    "HIGH": {
        "panel_fill": "#F5F9FF",
        "panel_edge": "#C8D9F2",
        "card_fill": "#ECF4FF",
        "card_edge": "#8FB0DB",
        "title": "#2E5E9E",
        "chip_fill": "#DCEBFF",
        "chip_edge": "#A7C2E8",
    },
    "LOW": {
        "panel_fill": "#FFF7F7",
        "panel_edge": "#E9CACA",
        "card_fill": "#FDEEEE",
        "card_edge": "#D7A2A2",
        "title": "#A34F4F",
        "chip_fill": "#FBE0E0",
        "chip_edge": "#E1B3B3",
    },
}
TEXT_COLOR = "#24333E"
MUTED_TEXT = "#5E6B75"

COND_PRETTY = {
    "S_high": ("高补贴", "非高补贴"),
    "L_long": ("长假期", "非长假期"),
    "C_relief": ("增强托育", "基础托育"),
    "I_high": ("较高收入", "低收入"),
    "G_help": ("有祖辈照护", "无祖辈照护"),
    "P_high": ("二孩/三孩", "较低孩次"),
}

PATH_TYPE_PRETTY = {
    "H1": "政策支持+家庭托底型",
    "H2": "政策支持+资源互补型",
    "L1": "结构性脆弱性锁定型",
    "L2": "单项补贴难以扭转型",
}

def short_metric(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"

def recipe_to_pretty_labels(expr: str) -> List[str]:
    lits = parse_recipe_expression(expr)
    labels=[]
    for cond, pos in lits:
        pos_lab, neg_lab = COND_PRETTY.get(cond, (cond, f"~{cond}"))
        labels.append(pos_lab if pos else neg_lab)
    return labels

def first_top_package(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    first = s.split(";")[0]
    return first.split(":")[0].strip()

def package_code_to_pretty(code: str) -> str:
    if not code or pd.isna(code):
        return ""
    try:
        parts = str(code).strip().split("-")
        mapping = {
            "S1": "低补贴", "S2": "中补贴", "S3": "高补贴",
            "L1": "短假期", "L2": "中假期", "L3": "长假期",
            "C1": "基础托育", "C2": "增强托育",
        }
        labs = [mapping.get(p.strip(), p.strip()) for p in parts if p.strip()]
        return "–".join(labs)
    except Exception:
        return str(code)

def concise_path_summary(labels: List[str], path_group: str) -> str:
    return ""

def wrap_to_lines(text: str, width: int = 22) -> List[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)

def path_type_pretty(path_id: str) -> str:
    return PATH_TYPE_PRETTY.get(str(path_id).strip(), "")

def select_representative_paths(sol_df: pd.DataFrame, mapping_df: pd.DataFrame, label: str, top_n: int = 2) -> pd.DataFrame:
    sub = sol_df.copy()
    sub = sub[(sub["label"] == label) & (sub["solution_type"] == "intermediate")].copy()
    sub = sub[sub["expression"].astype(str).str.strip().ne("")]
    sub = sub[sub["expression"].astype(str).str.strip().ne("(no solution)")]
    if sub.empty:
        return sub

    sub = sub.sort_values(["raw_coverage", "consistency", "freq"], ascending=[False, False, False])
    sub = sub.drop_duplicates(subset=["expression"], keep="first").head(top_n).copy()

    if mapping_df is not None and not mapping_df.empty:
        m = mapping_df.copy()
        merge_cols = [c for c in ["label", "solution_type", "expression", "implicant"] if c in m.columns and c in sub.columns]
        if merge_cols:
            sub = sub.merge(
                m[[c for c in m.columns if c in merge_cols or c in [
                    "contexts_n", "packages_n", "top5_packages_by_delta", "top5_packages_by_mu_hat",
                    "recipe_n_cases", "contexts_list", "packages_list"
                ]]],
                on=merge_cols,
                how="left",
            )

    prefix = "H" if label.upper() == "HIGH" else "L"
    sub = sub.reset_index(drop=True)
    sub["path_id"] = [f"{prefix}{i+1}" for i in range(len(sub))]
    sub["pretty_labels"] = sub["expression"].apply(recipe_to_pretty_labels)
    if "top5_packages_by_delta" in sub.columns:
        sub["top_package"] = sub["top5_packages_by_delta"].apply(first_top_package)
    elif "top5_packages_by_mu_hat" in sub.columns:
        sub["top_package"] = sub["top5_packages_by_mu_hat"].apply(first_top_package)
    else:
        sub["top_package"] = ""
    sub["top_package_pretty"] = sub["top_package"].apply(package_code_to_pretty)
    sub["path_type_pretty"] = sub["path_id"].apply(path_type_pretty)
    sub["summary_text"] = ""
    return sub

def _chip_width(text: str, base: float = 0.022, per_char: float = 0.012, min_w: float = 0.095) -> float:
    return max(min_w, base + per_char * len(text))

def _draw_chip(ax, x: float, y: float, text: str, style: dict, fontsize: float = 10.2):
    w = _chip_width(text)
    h = 0.048
    chip = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=style["chip_edge"],
        facecolor=style["chip_fill"],
        transform=ax.transAxes,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(chip)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=TEXT_COLOR, transform=ax.transAxes, zorder=4)
    return w, h

def draw_path_cards_figure(high_cards: pd.DataFrame, low_cards: pd.DataFrame, out_png: str):
    fig = plt.figure(figsize=(13.8, 8.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    panels = [
        ("HIGH", "高评价路径", "较强支持与家庭托底可以共同导向高评价", high_cards, 0.05),
        ("LOW",  "低评价锁定路径", "脆弱性条件更容易形成锁定结构", low_cards, 0.525),
    ]

    panel_y = 0.06
    panel_w = 0.42
    panel_h = 0.88

    for key, title, subtitle, cards, x0 in panels:
        style = CARD_STYLE[key]
        panel = FancyBboxPatch(
            (x0, panel_y), panel_w, panel_h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1.15, edgecolor=style["panel_edge"], facecolor=style["panel_fill"],
            transform=ax.transAxes, zorder=1
        )
        ax.add_patch(panel)

        ax.text(x0 + 0.025, panel_y + panel_h - 0.055, title,
                ha="left", va="center", fontsize=17.2, color=style["title"],
                fontweight="bold", transform=ax.transAxes)
        ax.text(x0 + 0.025, panel_y + panel_h - 0.093, subtitle,
                ha="left", va="center", fontsize=9.2, color=MUTED_TEXT,
                transform=ax.transAxes)

        if cards is None or cards.empty:
            ax.text(x0 + panel_w/2, panel_y + panel_h/2, "无可展示的中间解路径",
                    ha="center", va="center", fontsize=12.5, color=MUTED_TEXT, transform=ax.transAxes)
            continue

        n = len(cards)
        inner_top = panel_y + panel_h - 0.15
        gap = 0.035
        usable_h = panel_h - 0.19 - gap * (n - 1)
        card_h = usable_h / max(n, 1)
        card_h = min(card_h, 0.33)
        card_h = max(card_h, 0.30)

        for i, row in enumerate(cards.itertuples()):
            cy = inner_top - (i + 1) * card_h - i * gap
            cx = x0 + 0.02
            cw = panel_w - 0.04

            card = FancyBboxPatch(
                (cx, cy), cw, card_h,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                linewidth=1.02, edgecolor=style["card_edge"], facecolor=style["card_fill"],
                transform=ax.transAxes, zorder=2
            )
            ax.add_patch(card)

            ax.text(cx + 0.018, cy + card_h - 0.041, f"路径 {row.path_id}",
                    ha="left", va="center", fontsize=13.8, color=style["title"],
                    fontweight="bold", transform=ax.transAxes, zorder=4)

            type_text = getattr(row, "path_type_pretty", "")
            if type_text:
                w = max(0.13, 0.018 + 0.0105 * len(type_text))
                h = 0.042
                tx = cx + cw - w - 0.018
                ty = cy + card_h - 0.062
                tag = FancyBboxPatch(
                    (tx, ty), w, h,
                    boxstyle="round,pad=0.003,rounding_size=0.012",
                    linewidth=0.9, edgecolor=style["chip_edge"], facecolor=style["chip_fill"],
                    transform=ax.transAxes, clip_on=False, zorder=3
                )
                ax.add_patch(tag)
                ax.text(tx + w/2, ty + h/2, type_text, ha="center", va="center",
                        fontsize=8.8, color=TEXT_COLOR, transform=ax.transAxes, zorder=4)

            metric_line = (
                f"一致性 {short_metric(row.consistency)}  ｜  覆盖度 {short_metric(row.raw_coverage)}"
            )
            ax.text(cx + 0.018, cy + card_h - 0.083, metric_line,
                    ha="left", va="center", fontsize=9.9, color=TEXT_COLOR,
                    transform=ax.transAxes, zorder=4)

            chip_labels = list(getattr(row, "pretty_labels", []))[:4]
            chip_x = cx + 0.018
            chip_y = cy + card_h - 0.168
            row_h = 0.056
            current_x = chip_x
            current_y = chip_y
            rows_used = 1 if chip_labels else 0
            for lab in chip_labels:
                est_w = _chip_width(lab, base=0.018, per_char=0.010, min_w=0.09)
                if current_x + est_w > cx + cw - 0.02:
                    current_x = chip_x
                    current_y -= row_h
                    rows_used += 1
                w, _ = _draw_chip(ax, current_x, current_y, lab, style, fontsize=9.0)
                current_x += w + 0.012

            # Place bottom explanatory text dynamically below the last chip row
            bottom_block_top = (chip_y - (rows_used - 1) * row_h) - 0.038
            bottom_text_y = max(cy + 0.040, bottom_block_top)

            pkg_line = getattr(row, "top_package_pretty", "")
            if pkg_line:
                pkg_lines = wrap_to_lines(f"代表政策包：{pkg_line}", width=18)
                for j, line in enumerate(pkg_lines[:2]):
                    ax.text(cx + 0.018, bottom_text_y - j*0.028, line,
                            ha="left", va="center", fontsize=8.8, color=MUTED_TEXT,
                            transform=ax.transAxes, zorder=4)
                context_y = bottom_text_y - 0.034 - (len(pkg_lines[:2]) - 1) * 0.028
            else:
                context_y = bottom_text_y

            if hasattr(row, "contexts_n") and not pd.isna(row.contexts_n):
                ctx_text = f"覆盖 {int(row.contexts_n)} 类家庭处境"
                ax.text(cx + 0.018, context_y, ctx_text,
                        ha="left", va="center", fontsize=8.8, color=MUTED_TEXT,
                        transform=ax.transAxes, zorder=4)

    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = load_config()
    clean_fse_long_path = os.path.join(DATA_FINAL, "clean_fse_long_main.csv")
    if not os.path.exists(clean_fse_long_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_fse_long_main.csv")
        if os.path.exists(alt):
            clean_fse_long_path = alt
    observed_summary_path = os.path.join(DATA_FINAL, "scenario_observed_summary.csv")
    if not os.path.exists(observed_summary_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_observed_summary.csv")
        if os.path.exists(alt):
            observed_summary_path = alt

    # Optional: Module-B by-context table for mapping
    b3_path = os.path.join(OUT, "B3_packages_by_context_12x18.csv")
    b3_table = None
    if os.path.exists(b3_path):
        try:
            b3_table = pd.read_csv(b3_path)
        except Exception:
            b3_table = None

    # Build 216 scenarios and FE prediction
    full = build_full_scenarios()
    beta, alpha_bar, design_vec = fit_within_fe(clean_fse_long_path)
    preds=[]
    for _,r in full.iterrows():
        x = design_vec(int(r.P),int(r.I),int(r.G),int(r.S),int(r.L),int(r.C))
        preds.append(float(alpha_bar + x @ beta))
    full["mean_rating_pred"] = preds
    full.to_csv(os.path.join(OUT, "C0_cases_full216_pred.csv"), index=False, encoding="utf-8-sig")

    # Load observed summary (if exists)
    obs_df = None
    if os.path.exists(observed_summary_path):
        obs_df = pd.read_csv(observed_summary_path)
        # harmonize scenario_id
        if "scenario_id" not in obs_df.columns:
            # try build from P,I,G,S,L,Cdim
            cols = obs_df.columns
            cmap = {"C":"Cdim"} if "Cdim" in cols and "C" not in cols else {}
            # fall back: use first matching
            if all(k in cols for k in ["P","I","G","S","L"]) and ("Cdim" in cols or "C" in cols):
                if "C" in cols and "Cdim" not in cols:
                    obs_df["Cdim"] = obs_df["C"]
                obs_df["scenario_id"] = obs_df.apply(lambda r: f"{int(r.P)}{int(r.I)}{int(r.G)}{int(r.S)}{int(r.L)}{int(r.Cdim)}", axis=1)
        # pick columns
        # expected: mean_rating (or mean_rating_obs), n_obs, pr_ge8
        rename={}
        if "mean_rating" in obs_df.columns and "mean_rating_obs" not in obs_df.columns:
            rename["mean_rating"]="mean_rating_obs"
        if "n" in obs_df.columns and "n_obs" not in obs_df.columns:
            rename["n"]="n_obs"
        if rename:
            obs_df = obs_df.rename(columns=rename)

        # Try compute pr_ge8_obs if not present but have count_ge8 / n_obs etc.
        if "pr_ge8_obs" not in obs_df.columns:
            if "count_ge8" in obs_df.columns and "n_obs" in obs_df.columns:
                obs_df["pr_ge8_obs"] = obs_df["count_ge8"] / obs_df["n_obs"].replace(0, np.nan)
            elif "share_ge8" in obs_df.columns:
                obs_df["pr_ge8_obs"] = obs_df["share_ge8"]
        obs_df.to_csv(os.path.join(OUT, "C0_cases_observed.csv"), index=False, encoding="utf-8-sig")

    # Merge
    merged = full.copy()
    if obs_df is not None:
        keep_cols = [c for c in ["scenario_id","mean_rating_obs","n_obs","pr_ge8_obs"] if c in obs_df.columns]
        merged = merged.merge(obs_df[keep_cols], on="scenario_id", how="left")
    else:
        merged["mean_rating_obs"] = np.nan
        merged["n_obs"] = np.nan
        merged["pr_ge8_obs"] = np.nan


    # Ensure optional columns exist
    for col in ["mean_rating_obs","n_obs","pr_ge8_obs"]:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = fuzzy_memberships_from_levels(merged)
    merged.to_csv(os.path.join(OUT, "C0_cases_merged.csv"), index=False, encoding="utf-8-sig")

    # Outcome sources
    outcome_rows=[]
    # MAIN outcome: observed mean if available; else fall back to pred
    has_obs = merged["mean_rating_obs"].notna().sum() > 0
    y_main_raw = "mean_rating_obs" if has_obs else "mean_rating_pred"
    outcome_rows.append({"outcome":"MAIN_high", "y_raw":y_main_raw})

    # Robustness: pred
    outcome_rows.append({"outcome":"ROBUST_pred_high", "y_raw":"mean_rating_pred"})

    # Optional: pr_ge8
    if merged["pr_ge8_obs"].notna().sum() > 0:
        outcome_rows.append({"outcome":"ROBUST_pr_ge8_high", "y_raw":"pr_ge8_obs"})

    pd.DataFrame(outcome_rows).to_csv(os.path.join(OUT, "C6_outcome_comparison_summary.csv"), index=False, encoding="utf-8-sig")

    if "n_obs" in merged.columns and merged["n_obs"].notna().any():
        w_obs = merged["n_obs"].fillna(0.0).values.astype(float)
        # for predicted-only runs, avoid all-zero weights by fallback to 1
        w_pred = np.where(w_obs>0, w_obs, 1.0)
    else:
        w_obs = np.ones(len(merged), dtype=float)
        w_pred = np.ones(len(merged), dtype=float)

    oc = cfg.get("moduleC_calibration", {})
    full_out0 = float(oc.get("full_out", 6.0))
    cross0    = float(oc.get("crossover", 7.0))
    full_in0  = float(oc.get("full_in", 8.0))

    cons_thr = float(cfg.get("moduleC_consistency_threshold", DEFAULT_CONS_THR))
    freq_thr = float(cfg.get("moduleC_frequency_threshold", DEFAULT_FREQ_THR))
    pri_thr  = float(cfg.get("moduleC_PRI_threshold", 0.0))

    df_cases = merged.copy()
    w_main = w_obs if y_main_raw=="mean_rating_obs" else w_pred

    y_main_cal = outcome_with_prediction_fill(df_cases, y_main_raw)
    Y_high = calibrate_direct(y_main_cal, full_out0, cross0, full_in0)
    df_cases["Y_high"] = Y_high
    df_cases["Y_low"] = 1 - Y_high

    nec_high = necessary_conditions(df_cases, "Y_high", CONDS, w=w_main)
    nec_low  = necessary_conditions(df_cases, "Y_low",  CONDS, w=w_main)
    nec_high.to_csv(os.path.join(OUT, "C1_necessary_conditions_high.csv"), index=False, encoding="utf-8-sig")
    nec_low.to_csv(os.path.join(OUT, "C1_necessary_conditions_low.csv"), index=False, encoding="utf-8-sig")

    tt_high = truth_table_fs(df_cases, CONDS, y=df_cases["Y_high"].values, w=w_main, cons_thr=cons_thr, freq_thr=freq_thr, pri_thr=pri_thr)
    tt_low  = truth_table_fs(df_cases, CONDS, y=df_cases["Y_low"].values,  w=w_main, cons_thr=cons_thr, freq_thr=freq_thr, pri_thr=pri_thr)
    tt_high.to_csv(os.path.join(OUT, "C2_truth_table_high.csv"), index=False, encoding="utf-8-sig")
    tt_low.to_csv(os.path.join(OUT, "C3_truth_table_low.csv"), index=False, encoding="utf-8-sig")

    sol_high, sol_high_overall = solve(tt_high, df_cases, CONDS, y=df_cases["Y_high"].values, w=w_main,
                                       cons_thr=cons_thr, freq_thr=freq_thr, pri_thr=pri_thr, expect=DIR_EXPECT_HIGH, label="HIGH")
    sol_low, sol_low_overall = solve(tt_low, df_cases, CONDS, y=df_cases["Y_low"].values, w=w_main,
                                     cons_thr=cons_thr, freq_thr=freq_thr, pri_thr=pri_thr, expect=DIR_EXPECT_LOW, label="LOW")
    sol_high.to_csv(os.path.join(OUT, "C2_solutions_high.csv"), index=False, encoding="utf-8-sig")
    sol_high_overall.to_csv(os.path.join(OUT, "C2_solution_overall_high.csv"), index=False, encoding="utf-8-sig")
    sol_low.to_csv(os.path.join(OUT, "C3_solutions_low.csv"), index=False, encoding="utf-8-sig")
    sol_low_overall.to_csv(os.path.join(OUT, "C3_solution_overall_low.csv"), index=False, encoding="utf-8-sig")

    full_in_list = [full_in0-0.2, full_in0, full_in0+0.2]
    cross_list   = [cross0-0.2, cross0, cross0+0.2]
    full_out_list= [full_out0-0.2, full_out0, full_out0+0.2]
    cons_thr_list= [0.75, 0.80, 0.85]
    freq_thr_list= [1.0, 2.0]
    pri_thr_list = [0.0, 0.5] if pri_thr>0 else [0.0]
    sens = sensitivity_grid(df_cases, y_raw_col=y_main_raw, w=w_main,
                            full_in_list=full_in_list, cross_list=cross_list, full_out_list=full_out_list,
                            cons_thr_list=cons_thr_list, freq_thr_list=freq_thr_list, pri_thr_list=pri_thr_list)
    sens.to_csv(os.path.join(OUT, "C4_calibration_sensitivity_grid.csv"), index=False, encoding="utf-8-sig")

    recipe_rows = pd.concat([sol_high, sol_low], axis=0, ignore_index=True)
    mapping = summarize_recipe_mapping(df_cases, recipe_rows, b3_table=b3_table)
    mapping.to_csv(os.path.join(OUT, "C5_recipe_to_packages_contexts.csv"), index=False, encoding="utf-8-sig")

    high_cards = select_representative_paths(sol_high, mapping, label="HIGH", top_n=2)
    low_cards = select_representative_paths(sol_low, mapping, label="LOW", top_n=2)
    selected_cards = pd.concat([high_cards, low_cards], axis=0, ignore_index=True)
    if not selected_cards.empty:
        keep_cols = [c for c in [
            "label", "path_id", "solution_type", "expression", "implicant",
            "consistency", "raw_coverage", "freq", "contexts_n", "packages_n",
            "top_package", "contexts_list", "packages_list"
        ] if c in selected_cards.columns]
        selected_cards[keep_cols].to_csv(
            os.path.join(OUT, "C7_selected_path_cards.csv"),
            index=False, encoding="utf-8-sig"
        )
    draw_path_cards_figure(
        high_cards=high_cards,
        low_cards=low_cards,
        out_png=os.path.join(OUT, "C7_qca_path_cards_high_low.png"),
    )

    print(f"[OK] Module C v6 done. MAIN outcome: {y_main_raw}. Outputs -> {OUT}")

if __name__ == "__main__":
    main()
