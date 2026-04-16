# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FINAL = os.path.join(ROOT, "data", "final")
OUT        = os.path.join(ROOT, "outputs", "fse", "main")
CFG        = os.path.join(ROOT, "config", "config.json")
os.makedirs(OUT, exist_ok=True)

LEVELS = {
    "P":    [1, 2, 3],   # parity constraint / family stage
    "I":    [1, 2],      # income
    "G":    [1, 2],      # grandparent help
    "S":    [1, 2, 3],   # subsidy
    "L":    [1, 2, 3],   # leave
    "Cdim": [1, 2],      # childcare copay (2 = lower copay / more relief)
}

BASELINE = {"S": 1, "L": 1, "Cdim": 1}  
HHH      = {"S": 3, "L": 3, "Cdim": 2}     


PAPER_PALETTE = ["#D93F49", "#E28187", "#EBBFC2", "#D5E1E3", "#AFC9CF", "#8FB4BE"]
TEXT_COLOR = "#2F3A40"
GRID_COLOR = "#D9E3E6"
LINE_ZERO_COLOR = "#6A8E98"

FAMILY_COLORS = {
    "P": PAPER_PALETTE[0],
    "I": PAPER_PALETTE[1],
    "G": PAPER_PALETTE[2],
    "S": PAPER_PALETTE[0],
    "L": PAPER_PALETTE[1],
    "Cdim": PAPER_PALETTE[5],
}

LEAVE_LEVEL_COLORS = {1: PAPER_PALETTE[0], 2: PAPER_PALETTE[1], 3: PAPER_PALETTE[5]}
PACKAGE_COLOR_BASES = {"S": PAPER_PALETTE[0], "L": PAPER_PALETTE[1], "Cdim": PAPER_PALETTE[5]}
B3_COOL_BAR_PALETTE = ["#DDEBED", PAPER_PALETTE[3], PAPER_PALETTE[4], PAPER_PALETTE[5]]
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_diverging_red_positive",
    [PAPER_PALETTE[5], PAPER_PALETTE[3], "#F7F8F8", PAPER_PALETTE[2], PAPER_PALETTE[0]],
    N=256,
)

CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "PingFang SC", "Heiti SC", "WenQuanYi Zen Hei", "Arial Unicode MS"
]

def _setup_chinese_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CHINESE_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

_setup_chinese_font()

def _blend_hex(c1: str, c2: str, w: float = 0.5) -> str:
    w = float(np.clip(w, 0.0, 1.0))
    a = np.array(to_rgb(c1))
    b = np.array(to_rgb(c2))
    return to_hex((1 - w) * a + w * b)

def _mix_hex(colors: list[str], weights: list[float] | np.ndarray) -> str:
    arr = np.array([to_rgb(c) for c in colors], dtype=float)
    w = np.asarray(weights, dtype=float)
    if np.allclose(w.sum(), 0.0):
        w = np.ones(len(colors), dtype=float)
    w = w / w.sum()
    return to_hex(np.sum(arr * w[:, None], axis=0))

def _style_axes(ax, grid_axis: str | None = "y"):
    ax.set_facecolor("white")
    if grid_axis in {"x", "y", "both"}:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linestyle=":", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_blend_hex(TEXT_COLOR, "#FFFFFF", 0.55))
    ax.spines["bottom"].set_color(_blend_hex(TEXT_COLOR, "#FFFFFF", 0.55))
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)

def _package_generosity_score(S: int, L: int, Cdim: int) -> float:
    return ((S - 1) / 2.0 + (L - 1) / 2.0 + (Cdim - 1) / 1.0) / 3.0

def package_color(S: int, L: int, Cdim: int) -> str:
    score = _package_generosity_score(S, L, Cdim)
    weights = np.array([0.55 + 0.25 * (S - 1), 0.45 + 0.22 * (L - 1), 0.40 + 0.35 * (Cdim - 1)], dtype=float)
    base = _mix_hex([PACKAGE_COLOR_BASES["S"], PACKAGE_COLOR_BASES["L"], PACKAGE_COLOR_BASES["Cdim"]], weights)
    return _blend_hex(base, "#FFFFFF", 0.36 - 0.18 * score)

def package_colors(df_pkg: pd.DataFrame) -> list[str]:
    return [package_color(int(r.S), int(r.L), int(r.Cdim)) for r in df_pkg.itertuples()]

def package_color_cool_bar(S: int, L: int, Cdim: int) -> str:
    """Softer cool-toned colors for B3 package ranking bars."""
    score = _package_generosity_score(S, L, Cdim)
    if score <= 0.50:
        t = score / 0.50
        base = _blend_hex(B3_COOL_BAR_PALETTE[0], B3_COOL_BAR_PALETTE[1], t)
    else:
        t = (score - 0.50) / 0.50
        base = _blend_hex(B3_COOL_BAR_PALETTE[1], B3_COOL_BAR_PALETTE[3], t)
    # keep the bars gentle rather than saturated
    soften = 0.22 if Cdim == 2 else 0.32
    return _blend_hex(base, "#FFFFFF", soften)

def package_cool_bar_colors(df_pkg: pd.DataFrame) -> list[str]:
    return [package_color_cool_bar(int(r.S), int(r.L), int(r.Cdim)) for r in df_pkg.itertuples()]

def _package_display_label(S: int, L: int, Cdim: int) -> str:
    subsidy = {1: "低补贴", 2: "中补贴", 3: "高补贴"}[int(S)]
    leave = {1: "短假期", 2: "中假期", 3: "长假期"}[int(L)]
    care = {1: "基础托育", 2: "增强托育"}[int(Cdim)]
    return f"{subsidy}–{leave}–{care}"


def _representative_package_keys() -> list[tuple[int, int, int]]:
    return [
        (1, 1, 1),  # baseline
        (1, 2, 1),  # leave-focused
        (2, 1, 1),  # subsidy-focused
        (1, 1, 2),  # childcare-only
        (2, 2, 1),  # balanced mid without childcare enhancement
        (2, 2, 2),  # balanced mid with childcare enhancement
        (3, 2, 2),  # strong integrated
        (3, 3, 2),  # strongest package
    ]


def select_representative_packages(df_pkg: pd.DataFrame) -> pd.DataFrame:
    wanted = _representative_package_keys()
    order = {k: i for i, k in enumerate(wanted)}
    sub = df_pkg[df_pkg[["S", "L", "Cdim"]].apply(lambda r: (int(r["S"]), int(r["L"]), int(r["Cdim"])), axis=1).isin(wanted)].copy()
    sub["_ord"] = sub[["S", "L", "Cdim"]].apply(lambda r: order[(int(r["S"]), int(r["L"]), int(r["Cdim"]))], axis=1)
    sub = sub.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    sub["display_label"] = sub.apply(lambda r: _package_display_label(int(r["S"]), int(r["L"]), int(r["Cdim"])), axis=1)
    sub["profile"] = [
        "低支持参照组合",
        "延长假期",
        "提高补贴",
        "增强托育",
        "中补贴–中假期–基础托育",
        "中补贴–中假期–增强托育",
        "高补贴–中假期–增强托育",
        "高补贴–长假期–增强托育",
    ][: len(sub)]
    return sub


def plot_package_bar_ci_full(df_pkg: pd.DataFrame, out_png: str, title: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_pkg))
    colors = package_cool_bar_colors(df_pkg)
    edge_colors = [_blend_hex(c, TEXT_COLOR, 0.18) for c in colors]
    ax.bar(x, df_pkg["mu_hat"], color=colors, edgecolor=edge_colors, linewidth=0.8)
    yerr = np.vstack([df_pkg["mu_hat"] - df_pkg["ci_low"], df_pkg["ci_high"] - df_pkg["mu_hat"]])
    ax.errorbar(x, df_pkg["mu_hat"], yerr=yerr, fmt="none", ecolor=_blend_hex(TEXT_COLOR, B3_COOL_BAR_PALETTE[-1], 0.35), elinewidth=1.0, capsize=3, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(df_pkg["label"], rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("平均预测评分（95%置信区间）")
    if title:
        ax.set_title(title)
    _style_axes(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)



def _prepare_all_packages_for_plot(df_pkg: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    sub = df_pkg.copy()
    sub["display_label"] = sub.apply(lambda r: _package_display_label(int(r["S"]), int(r["L"]), int(r["Cdim"])), axis=1)
    sub = sub.sort_values(["delta_vs_baseline", "mu_hat", "S", "L", "Cdim"], ascending=[ascending, ascending, True, True, True]).reset_index(drop=True)
    return sub

def _barh_with_ci(ax, sub: pd.DataFrame, title: str, xlim: tuple[float, float]):
    y = np.arange(len(sub))
    colors = package_cool_bar_colors(sub)
    edge_colors = [_blend_hex(c, TEXT_COLOR, 0.18) for c in colors]

    ax.barh(
        y,
        sub["delta_vs_baseline"],
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.9,
        height=0.62,
    )
    xerr = np.vstack([
        sub["delta_vs_baseline"] - sub["delta_ci_low"],
        sub["delta_ci_high"] - sub["delta_vs_baseline"],
    ])
    ax.errorbar(
        sub["delta_vs_baseline"],
        y,
        xerr=xerr,
        fmt="none",
        ecolor=_blend_hex(TEXT_COLOR, B3_COOL_BAR_PALETTE[-1], 0.35),
        elinewidth=1.0,
        capsize=2.5,
        alpha=0.95,
        zorder=3,
    )

    label_offset = 0.015 * (xlim[1] - xlim[0])
    for yi, row in zip(y, sub.itertuples()):
        ax.text(
            float(row.delta_ci_high) + label_offset,
            yi,
            f"Δ={row.delta_vs_baseline:.2f}",
            va="center",
            ha="left",
            fontsize=7.8,
            color=TEXT_COLOR,
        )

    ax.axvline(0, linestyle="--", linewidth=1.3, color=LINE_ZERO_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["display_label"], fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=12, pad=8)
    _style_axes(ax, grid_axis="x")

def plot_package_all_dual_orders(df_pkg: pd.DataFrame, out_png: str, title: str | None = None):
    """
    主文图1改版：18类政策组合拆分为左右双面板。
    左图展示评分增益较低的 9 类（升序），右图展示评分增益较高的 9 类（降序）。
    """
    asc_all = _prepare_all_packages_for_plot(df_pkg, ascending=True)
    desc_all = _prepare_all_packages_for_plot(df_pkg, ascending=False)

    n_total = len(asc_all)
    n_left = int(np.ceil(n_total / 2))
    n_right = n_total - n_left
    if n_right == 0:
        n_right = n_left

    # 左图：整体升序后的前 9 类（较低增益部分）
    left = asc_all.iloc[:n_left].reset_index(drop=True)
    # 右图：整体降序后的前 9 类（较高增益部分）
    right = desc_all.iloc[:n_right].reset_index(drop=True)

    xmin = min(-0.02, float(df_pkg["delta_ci_low"].min()) - 0.02)
    xmax = float(df_pkg["delta_ci_high"].max()) + 0.12

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.6), sharex=True)
    _barh_with_ci(axes[0], left, "升序（9类）", (xmin, xmax))
    _barh_with_ci(axes[1], right, "降序（9类）", (xmin, xmax))

    axes[0].set_xlabel("评分增益 Δ（95%置信区间）", fontsize=10.5)
    axes[1].set_xlabel("评分增益 Δ（95%置信区间）", fontsize=10.5)
    axes[0].set_ylabel("政策组合", fontsize=10.5)
    axes[1].set_ylabel("政策组合", fontsize=10.5)

    if title:
        fig.suptitle(title, fontsize=13, y=0.98, color=TEXT_COLOR)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.92, wspace=0.28)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

def plot_package_overall_representatives(df_pkg: pd.DataFrame, out_png: str, title: str | None = None):
    # 保留旧函数接口，但主文图改为 18 类政策组合双面板排序图
    plot_package_all_dual_orders(df_pkg, out_png, title=title)

def load_config() -> dict:
    if os.path.exists(CFG):
        with open(CFG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def _cluster_robust_cov(X: np.ndarray, u: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Arellano-style cluster robust covariance."""
    XtX_inv = np.linalg.inv(X.T @ X)
    S = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        idx = (groups == g)
        Xg = X[idx]
        ug = u[idx].reshape(-1, 1)
        v = Xg.T @ ug
        S += v @ v.T
    return XtX_inv @ S @ XtX_inv

def _dummies_fixed(df: pd.DataFrame, col: str, drop_first: bool = True) -> pd.DataFrame:
    """Create stable dummies even if df[col] is constant."""
    cats = LEVELS[col]
    x = pd.Categorical(df[col].astype(int).values, categories=cats, ordered=True)
    return pd.get_dummies(x, prefix=col, drop_first=drop_first, dtype=float)

def build_design(df: pd.DataFrame, spec: dict, ref: dict | None = None) -> tuple[np.ndarray, list[str], dict]:
    """
    spec:
      base_vars: list[str]
      interactions: list[tuple[str,str]]
      triple: list[tuple[str,str,str]]
    """
    D = {}
    for v in spec["base_vars"]:
        D[v] = _dummies_fixed(df, v, drop_first=True)

    Xdf = pd.concat([D[v] for v in spec["base_vars"]], axis=1)

    # pairwise interactions
    for a, b in spec.get("interactions", []):
        for ca in D[a].columns:
            for cb in D[b].columns:
                Xdf[f"{ca}:{cb}"] = D[a][ca].values * D[b][cb].values

    # triple interactions
    for a, b, c in spec.get("triple", []):
        for ca in D[a].columns:
            for cb in D[b].columns:
                for cc in D[c].columns:
                    Xdf[f"{ca}:{cb}:{cc}"] = D[a][ca].values * D[b][cb].values * D[c][cc].values

    # Align columns if ref given
    if ref is not None:
        for col in ref["design_cols"]:
            if col not in Xdf.columns:
                Xdf[col] = 0.0
        Xdf = Xdf[ref["design_cols"]]
        return Xdf.values.astype(float), list(Xdf.columns), ref

    ref = {"design_cols": list(Xdf.columns)}
    return Xdf.values.astype(float), list(Xdf.columns), ref

# ----------------------------
# B0 design diagnostics
# ----------------------------
def design_balance_by_t(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in sorted(df["t"].unique()):
        d = df[df["t"] == t]
        for v, levels in LEVELS.items():
            vc = d[v].value_counts(dropna=False).to_dict()
            for lv in levels:
                rows.append({"t": int(t), "var": v, "level": lv, "count": int(vc.get(lv, 0))})
    out = pd.DataFrame(rows)
    out["share_within_t_var"] = out.groupby(["t", "var"])["count"].transform(lambda x: x / x.sum() if x.sum() else 0.0)
    return out

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V for two categorical variables."""
    tab = pd.crosstab(x, y)
    n = tab.values.sum()
    if n == 0:
        return np.nan
    chi2 = ((tab - (tab.sum(axis=1).values.reshape(-1,1) @ tab.sum(axis=0).values.reshape(1,-1) / n))**2 /
            (tab.sum(axis=1).values.reshape(-1,1) @ tab.sum(axis=0).values.reshape(1,-1) / n + 1e-12)).values.sum()
    r, k = tab.shape
    denom = n * (min(r-1, k-1) if min(r-1, k-1) > 0 else 1)
    return float(np.sqrt(chi2 / denom))

def orthogonality_checks(df: pd.DataFrame) -> pd.DataFrame:
    vars_ = ["P","I","G","S","L","Cdim"]
    rows=[]
    # use all rows (pooled)
    for a,b in itertools.combinations(vars_,2):
        rows.append({"var_a":a, "var_b":b, "cramers_v": cramers_v(df[a], df[b])})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)

# ----------------------------
# B1: Between OLS by t
# ----------------------------
def between_additive_cluster(df: pd.DataFrame, t: int) -> pd.DataFrame:
    d = df[df["t"] == t].dropna(subset=["rating","rid","P","I","G","S","L","Cdim"]).copy()
    y = d["rating"].values.astype(float)
    spec = {"base_vars":["P","I","G","S","L","Cdim"], "interactions":[], "triple":[]}
    X, cols, _ = build_design(d, spec, ref=None)
    X = np.column_stack([np.ones(len(d)), X])
    cols = ["Intercept"] + cols
    beta = _ols_beta(X, y)
    u = y - X @ beta
    groups = pd.Categorical(d["rid"]).codes
    V = _cluster_robust_cov(X, u, groups)
    se = np.sqrt(np.diag(V))
    out = pd.DataFrame({"t": int(t), "term": cols, "coef": beta, "se": se})
    out["ci_low"] = out["coef"] - 1.96*out["se"]
    out["ci_high"]= out["coef"] + 1.96*out["se"]
    return out

def plot_policy_terms_by_t(b1: pd.DataFrame, out_png: str):
    # Focus on policy terms (S,L,Cdim) and key context G/I/P for one integrated coefficient trace figure
    keep = []
    family_order = ["S", "L", "Cdim", "G", "I", "P"]
    for v in family_order:
        for lv in LEVELS[v][1:]:
            keep.append(f"{v}_{lv}")

    sub = b1[b1["term"].isin(keep)].copy()
    if sub.empty:
        return

    terms = [t for t in keep if t in set(sub["term"])]
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(terms) + 2.1))
    y_positions = np.arange(len(terms))

    for i, term in enumerate(terms):
        d = sub[sub["term"] == term].sort_values("t")
        fam = term.split("_")[0]
        color = FAMILY_COLORS.get(fam, PAPER_PALETTE[-1])
        line_color = _blend_hex(color, TEXT_COLOR, 0.15)
        ax.plot(d["coef"], np.full(len(d), i), color=_blend_hex(color, "#FFFFFF", 0.18), linewidth=2.0, alpha=0.95)
        ax.errorbar(
            d["coef"],
            np.full(len(d), i),
            xerr=1.96 * d["se"],
            fmt="o",
            color=line_color,
            ecolor=_blend_hex(color, TEXT_COLOR, 0.30),
            elinewidth=1.8,
            capsize=3,
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            alpha=0.95,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(terms)
    ax.axvline(0, linestyle="--", linewidth=1.5, color=LINE_ZERO_COLOR)
    ax.set_xlabel("Coefficient (between OLS), 95% CI; each dot = one t")
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# B2: Within FE (main spec)
# ----------------------------
def fe_within_cluster(df: pd.DataFrame, spec: dict):
    d = df.dropna(subset=["rating","rid"]+spec["base_vars"]).copy()
    y = d["rating"].values.astype(float)
    groups = pd.Categorical(d["rid"]).codes

    X, cols, ref = build_design(d, spec, ref=None)

    # within transform by rid
    y_bar = pd.Series(y).groupby(groups).transform("mean").values
    X_bar = pd.DataFrame(X).groupby(groups).transform("mean").values
    y_dm = y - y_bar
    X_dm = X - X_bar

    beta = _ols_beta(X_dm, y_dm)
    u = y_dm - X_dm @ beta
    V = _cluster_robust_cov(X_dm, u, groups)
    se = np.sqrt(np.diag(V))
    coef = pd.DataFrame({"term": cols, "coef": beta, "se": se})
    coef["ci_low"]  = coef["coef"] - 1.96*coef["se"]
    coef["ci_high"] = coef["coef"] + 1.96*coef["se"]

    # group fixed effects a_g = ybar_g - Xbar_g beta
    ybar_g = pd.Series(y).groupby(groups).mean().values
    Xbar_g = pd.DataFrame(X).groupby(groups).mean().values
    a_g = ybar_g - Xbar_g @ beta

    return coef, beta, ref, groups, a_g, d

def predict_cf(d_base: pd.DataFrame, spec: dict, ref: dict, beta: np.ndarray, groups: np.ndarray, a_g: np.ndarray,
               S: int, L: int, Cdim: int,
               P: int | None = None, I: int | None = None, G: int | None = None) -> np.ndarray:
    """Row-level predictions under counterfactual S/L/Cdim; optionally fix context P/I/G."""
    cf = d_base.copy()
    cf["S"] = int(S)
    cf["L"] = int(L)
    cf["Cdim"] = int(Cdim)
    if P is not None: cf["P"] = int(P)
    if I is not None: cf["I"] = int(I)
    if G is not None: cf["G"] = int(G)
    Xcf, _, _ = build_design(cf, spec, ref=ref)
    return a_g[groups] + Xcf @ beta

def respondent_means(yhat_rows: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Mean predicted rating per respondent."""
    return pd.Series(yhat_rows).groupby(groups).mean().values

def bootstrap_ci(means: np.ndarray, B: int = 300, seed: int = 42) -> tuple[float,float]:
    rng = np.random.default_rng(seed)
    n = len(means)
    idx = np.arange(n)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        boots[b] = float(np.mean(means[samp]))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

# ----------------------------
# Plotting helpers (policy slices)
# ----------------------------
def plot_policy_slices(df_pkg: pd.DataFrame, out_png: str):
    """主文图2：不同补贴–假期结构下，基础托育与增强托育的比较。"""
    mats = {}
    vmax = 0.0
    for Cdim in [1, 2]:
        piv = (
            df_pkg[df_pkg["Cdim"] == Cdim]
            .pivot_table(index="L", columns="S", values="delta_vs_baseline", aggfunc="mean")
            .reindex(index=[1, 2, 3], columns=[1, 2, 3])
        )
        mats[Cdim] = piv
        vals = piv.values.astype(float)
        vmax = max(vmax, float(np.nanmax(np.abs(vals))))
    vmax = max(vmax, 1e-8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(10.4, 4.6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.18)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

    titles = {1: "基础托育", 2: "增强托育"}
    for j, Cdim in enumerate([1, 2]):
        ax = axes[j]
        vals = mats[Cdim].values.astype(float)
        im = ax.imshow(vals, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")
        ax.set_title(titles[Cdim], fontsize=13, pad=8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["低补贴", "中补贴", "高补贴"], fontsize=9.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["短假期", "中假期", "长假期"], fontsize=9.5)
        if j == 0:
            ax.set_ylabel("假期强度", fontsize=10)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("补贴强度", fontsize=10)
        for i in range(vals.shape[0]):
            for k in range(vals.shape[1]):
                ax.text(k, i, f"{vals[i, k]:.2f}", ha="center", va="center", fontsize=8.6, color=TEXT_COLOR)
        ax.set_xticks(np.arange(-0.5, vals.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, vals.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        _style_axes(ax, grid_axis=None)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("评分增益 Δ", fontsize=10, labelpad=8, color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_COLOR)
    cbar.outline.set_edgecolor(_blend_hex(TEXT_COLOR, "#FFFFFF", 0.55))
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.16, top=0.90)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

def heatmap_hhh_lll(grid: pd.DataFrame, out_png: str):
    """主文图3：双层分面热力图；左右 panel 拆分祖辈照护状态。"""
    row_order = [1, 2, 3]  # 一孩/二孩/三孩
    income_order = [1, 2]  # 低收入/较高收入
    gp_order = [1, 2]      # 无祖辈照护/有祖辈照护

    mats = {}
    all_vals = []
    for G in gp_order:
        mat = np.full((len(row_order), len(income_order)), np.nan, dtype=float)
        for i, P in enumerate(row_order):
            for j, I in enumerate(income_order):
                sub = grid[(grid["P"] == P) & (grid["I"] == I) & (grid["G"] == G)]
                if not sub.empty:
                    v = float(sub["delta"].iloc[0])
                    mat[i, j] = v
                    if np.isfinite(v):
                        all_vals.append(v)
        mats[G] = mat

    vmax = float(np.nanmax(np.abs(all_vals))) if len(all_vals) else 1.0
    vmax = max(vmax, 1e-8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(9.2, 4.9))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.065], wspace=0.10)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

    titles = {1: "无祖辈照护", 2: "有祖辈照护"}
    xticklabels = ["低收入", "较高收入"]
    yticklabels = ["一孩", "二孩", "三孩"]

    max_pos = None
    if len(all_vals):
        best_v = -np.inf
        for G in gp_order:
            mat = mats[G]
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if np.isfinite(v) and v > best_v:
                        best_v = v
                        max_pos = (G, i, j)

    im = None
    for ax, G in zip(axes, gp_order):
        mat = mats[G]
        im = ax.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP, norm=norm)
        ax.set_title(titles[G], fontsize=12.5, pad=8)
        ax.set_xticks(range(len(income_order)))
        ax.set_xticklabels(xticklabels, fontsize=9.4)
        ax.set_yticks(range(len(row_order)))
        if G == 1:
            ax.set_yticklabels(yticklabels, fontsize=9.4)
            ax.set_ylabel("孩次", fontsize=10)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("收入水平", fontsize=10)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt = f"{v:.2f}" if np.isfinite(v) else "NA"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.7, color=TEXT_COLOR)

        if max_pos is not None and max_pos[0] == G:
            _, i_max, j_max = max_pos
            ax.add_patch(plt.Rectangle((j_max-0.5, i_max-0.5), 1, 1, fill=False, edgecolor="#FFFFFF", linewidth=1.8))
            ax.text(j_max, i_max-0.38, "最高", ha="center", va="center", fontsize=8.2, color="#FFFFFF", fontweight="bold")

        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        _style_axes(ax, grid_axis=None)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("评分增益 Δ", fontsize=10, labelpad=8, color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=9, colors=TEXT_COLOR)
    cbar.outline.set_edgecolor(_blend_hex(TEXT_COLOR, "#FFFFFF", 0.55))

    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.16, top=0.90)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

def main():
    cfg = load_config()

    # Optional: clean previous outputs to avoid mixing versions
    if os.environ.get("CLEAN_OUTPUTS", "0") in ("1", "true", "True", "YES", "yes"):
        for fn in os.listdir(OUT):
            if fn.startswith(("B0_", "B1_", "B2_", "B3_", "B4_")):
                try:
                    os.remove(os.path.join(OUT, fn))
                except Exception:
                    pass

    df = pd.read_csv(os.path.join(DATA_FINAL, "clean_fse_long_main.csv"))
    if "C" in df.columns and "Cdim" not in df.columns:
        df = df.rename(columns={"C":"Cdim"})
    df = df.dropna(subset=["rid","t","rating","P","I","G","S","L","Cdim"]).copy()
    # ensure ints
    for c in ["t","P","I","G","S","L","Cdim"]:
        df[c] = df[c].astype(int)

    # -------------
    # B0 diagnostics
    # -------------
    bal = design_balance_by_t(df)
    bal.to_csv(os.path.join(OUT,"B0_design_balance_by_t.csv"), index=False, encoding="utf-8-sig")
    ortho = orthogonality_checks(df)
    ortho.to_csv(os.path.join(OUT,"B0_cramersV_pairwise.csv"), index=False, encoding="utf-8-sig")

    # -------------
    # B1 between by t=1..6
    # -------------
    b1_all = []
    for t in sorted(df["t"].unique()):
        b1_all.append(between_additive_cluster(df, int(t)))
    b1 = pd.concat(b1_all, ignore_index=True)
    b1.to_csv(os.path.join(OUT,"B1_between_by_t_coefs.csv"), index=False, encoding="utf-8-sig")

    # policy term tracking by t
    policy_terms = []
    for v in ["S","L","Cdim"]:
        for lv in LEVELS[v][1:]:
            policy_terms.append(f"{v}_{lv}")
    b1_pol = b1[(b1["term"].isin(policy_terms)) & (b1["term"]!="Intercept")].copy()
    b1_pol.to_csv(os.path.join(OUT,"B1_between_policy_terms_by_t.csv"), index=False, encoding="utf-8-sig")
    plot_policy_terms_by_t(b1, os.path.join(OUT,"B1_between_policy_terms_by_t.png"))

    # -------------
    # B2 within FE main spec:
    #   - main effects: P/I/G/S/L/Cdim
    #   - policy complementarity: S×L, S×C, L×C, and S×L×C
    # -------------
    # 1) 主规格：用于 B2（主回归）+ B3（18政策包）
    spec_main = {
        "base_vars": ["P","I","G","S","L","Cdim"],
        "interactions": [("S","L"), ("S","Cdim"), ("L","Cdim")],
        "triple": [("S","L","Cdim")]
    }
    fe_coef, beta, ref, groups, a_g, d_base = fe_within_cluster(df, spec_main)
    fe_coef.to_csv(os.path.join(OUT,"B2_fe_main_coefs.csv"), index=False, encoding="utf-8-sig")

    # 2) 异质性扩展规格：只用于 B4（ΔHHH–LLL(p,i,g)）
    spec_het = {
        "base_vars": ["P","I","G","S","L","Cdim"],
        "interactions": [
            ("S","L"), ("S","Cdim"), ("L","Cdim"),
            ("P","S"), ("P","L"), ("P","Cdim"),
            ("I","S"), ("I","L"), ("I","Cdim"),
            ("G","S"), ("G","L"), ("G","Cdim"),
        ],
        "triple": [("S","L","Cdim")]
    }
    fe_coef_het, beta_het, ref_het, groups_het, a_g_het, d_base_het = fe_within_cluster(df, spec_het)
    fe_coef_het.to_csv(os.path.join(OUT,"B2_fe_heterogeneity_coefs.csv"), index=False, encoding="utf-8-sig")

    # -------------
    # B3 policy packages: mean + CI + delta vs baseline
    # respondent bootstrap for CI (fast, cross-platform)
    # -------------
    B = int(os.environ.get("BOOTSTRAP_B", "300"))
    seed = int(os.environ.get("BOOTSTRAP_SEED", "42"))

    pkgs=[]
    # baseline respondent means (for delta)
    y_base_rows = predict_cf(d_base, spec_main, ref, beta, groups, a_g, **BASELINE)
    base_means = respondent_means(y_base_rows, groups)

    for S in LEVELS["S"]:
        for L in LEVELS["L"]:
            for Cdim in LEVELS["Cdim"]:
                yhat_rows = predict_cf(d_base, spec_main, ref, beta, groups, a_g, S=S, L=L, Cdim=Cdim)
                means = respondent_means(yhat_rows, groups)
                mu = float(np.mean(means))
                ci_lo, ci_hi = bootstrap_ci(means, B=B, seed=seed)
                # delta vs baseline (respondent-level difference)
                delta_means = means - base_means
                delta = float(np.mean(delta_means))
                d_lo, d_hi = bootstrap_ci(delta_means, B=B, seed=seed+1)
                pkgs.append({
                    "S": int(S), "L": int(L), "Cdim": int(Cdim),
                    "label": f"S{S}-L{L}-C{Cdim}",
                    "mu_hat": mu, "ci_low": ci_lo, "ci_high": ci_hi,
                    "delta_vs_baseline": delta, "delta_ci_low": d_lo, "delta_ci_high": d_hi
                })
    pkg = pd.DataFrame(pkgs).sort_values("mu_hat", ascending=False).reset_index(drop=True)
    pkg.to_csv(os.path.join(OUT,"B3_packages_mean_ci_delta.csv"), index=False, encoding="utf-8-sig")
    pkg_main = select_representative_packages(pkg)
    pkg_main.to_csv(os.path.join(OUT, "B3_packages_representative_mainfigure.csv"), index=False, encoding="utf-8-sig")

    plot_package_bar_ci_full(pkg, os.path.join(OUT,"B3_packages_bar_ci_full.png"))
    plot_package_all_dual_orders(pkg, os.path.join(OUT,"B3_packages_bar_ci.png"))
    plot_policy_slices(pkg, os.path.join(OUT,"B3_policy_slices.png"))


    # -------------
    # B3 (plan-aligned): fixed-context policy package menu(s)
    #  - Research plan requires comparing policy packages under fixed family contexts (P,I,G),
    #    not only averaging over observed context distribution.
    # We output two canonical stress-test contexts used in the proposal:
    #   (P=1,I=1,G=1) first child, low income, no grandparent help
    #   (P=2,I=1,G=1) second child, low income, no grandparent help
    # -------------
    # Fixed contexts for *menu* outputs (recommended by research plan)
    fixed_contexts_menu = [(1,1,1), (2,1,1)]  # P,I,G

    # Full 12 contexts (P×I×G) for 12×18 table (downstream Module C/D)
    contexts_full = [(P,I,G) for P in LEVELS["P"] for I in LEVELS["I"] for G in LEVELS["G"]]

    all_ctx_rows = []

    for (P_fix, I_fix, G_fix) in contexts_full:

        # baseline under fixed context
        y_base_rows_ctx = predict_cf(d_base, spec_main, ref, beta, groups, a_g,
                                     P=P_fix, I=I_fix, G=G_fix, **BASELINE)
        base_means_ctx = respondent_means(y_base_rows_ctx, groups)

        rows=[]
        for S in LEVELS["S"]:
            for L in LEVELS["L"]:
                for Cdim in LEVELS["Cdim"]:
                    yhat_rows = predict_cf(d_base, spec_main, ref, beta, groups, a_g,
                                           P=P_fix, I=I_fix, G=G_fix, S=S, L=L, Cdim=Cdim)
                    means = respondent_means(yhat_rows, groups)
                    mu = float(np.mean(means))
                    ci_lo, ci_hi = bootstrap_ci(means, B=B, seed=seed)

                    delta_means = means - base_means_ctx
                    delta = float(np.mean(delta_means))
                    d_lo, d_hi = bootstrap_ci(delta_means, B=B, seed=seed+1)

                    r = {
                        "P": P_fix, "I": I_fix, "G": G_fix,
                        "S": int(S), "L": int(L), "Cdim": int(Cdim),
                        "label": f"S{S}-L{L}-C{Cdim}",
                        "mu_hat": mu, "ci_low": ci_lo, "ci_high": ci_hi,
                        "delta_vs_baseline": delta, "delta_ci_low": d_lo, "delta_ci_high": d_hi
                    }
                    rows.append(r)
                    all_ctx_rows.append(r)

        df_ctx = pd.DataFrame(rows).sort_values("mu_hat", ascending=False).reset_index(drop=True)
        # Save per-context menu outputs only for selected contexts (avoid clutter)
        if (P_fix, I_fix, G_fix) in fixed_contexts_menu:
            out_csv = os.path.join(OUT, f"B3_packages_mean_ci_delta_context_P{P_fix}I{I_fix}G{G_fix}.csv")
            df_ctx.to_csv(out_csv, index=False, encoding="utf-8-sig")

            plot_package_bar_ci_full(
                df_ctx,
                os.path.join(OUT, f"B3_packages_bar_ci_context_P{P_fix}I{I_fix}G{G_fix}.png"),
                title=f"固定家庭处境下的政策组合（P={P_fix}, I={I_fix}, G={G_fix}）",
            )

    # Full 12×18 table (P×I×G contexts × 18 packages): plan-aligned output for downstream modules C/D
    df_ctx_all = pd.DataFrame(all_ctx_rows)
    df_ctx_all.to_csv(os.path.join(OUT, "B3_packages_by_context_12x18.csv"), index=False, encoding="utf-8-sig")


    # -------------
    # B4 HHH-LLL by context (12 cells) + CI + heatmap
    # Fixed context evaluation: for each (P,I,G), set context fixed for everyone, compare HHH vs baseline LLL.
    # -------------
    grid_rows=[]
    for P in LEVELS["P"]:
        for I in LEVELS["I"]:
            for G in LEVELS["G"]:
                y_h = predict_cf(d_base_het, spec_het, ref_het, beta_het, groups_het, a_g_het, P=P, I=I, G=G, **HHH)
                y_l = predict_cf(d_base_het, spec_het, ref_het, beta_het, groups_het, a_g_het, P=P, I=I, G=G, **BASELINE)
                mh = respondent_means(y_h, groups_het)
                ml = respondent_means(y_l, groups_het)
                delta_means = mh - ml
                delta = float(np.mean(delta_means))
                lo, hi = bootstrap_ci(delta_means, B=B, seed=seed+7)
                grid_rows.append({"P":P,"I":I,"G":G,"delta":delta,"ci_low":lo,"ci_high":hi,"n_resp":int(len(delta_means))})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(OUT,"B4_hhh_lll_delta_by_context_ci.csv"), index=False, encoding="utf-8-sig")
    heatmap_hhh_lll(grid, os.path.join(OUT,"B4_hhh_lll_heatmap.png"))

if __name__ == "__main__":
    main()
