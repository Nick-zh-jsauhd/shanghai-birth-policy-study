#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model 3 visualization suite (v3)
--------------------------------
Presentation-oriented assets for the "Python charts + Canva layout" workflow.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


PALETTE = {
    "navy": "#2B4C61",
    "slate": "#4E6A7C",
    "ink": "#31424F",
    "red": "#D85C4A",
    "rose": "#E89B8E",
    "gold": "#D7A84A",
    "teal": "#6D9CAB",
    "mist": "#EAF0F2",
    "sand": "#F7F3EC",
    "grey": "#BBC7CF",
    "light_grey": "#DCE5EA",
    "grid": "#E8EEF1",
    "white": "#FFFFFF",
}


BUDGET_ORDER = ["tight", "moderate", "ample", "max"]
BUDGET_YI = {"tight": 11, "moderate": 22, "ample": 44, "max": 89}


def set_global_style() -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["axes.edgecolor"] = PALETTE["ink"]
    matplotlib.rcParams["axes.labelcolor"] = PALETTE["ink"]
    matplotlib.rcParams["xtick.color"] = PALETTE["ink"]
    matplotlib.rcParams["ytick.color"] = PALETTE["ink"]
    matplotlib.rcParams["text.color"] = PALETTE["ink"]
    matplotlib.rcParams["axes.titleweight"] = "bold"
    matplotlib.rcParams["axes.titlesize"] = 15
    matplotlib.rcParams["axes.labelsize"] = 12
    matplotlib.rcParams["legend.frameon"] = False
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.dpi"] = 320


def ensure_dirs(output_dir: str) -> Dict[str, str]:
    out = {
        "root": output_dir,
        "figures": os.path.join(output_dir, "figures"),
        "appendix": os.path.join(output_dir, "appendix"),
    }
    for path in out.values():
        os.makedirs(path, exist_ok=True)
    return out


def save_figure(fig: plt.Figure, folder: str, stem: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(folder, f"{stem}.{ext}"))
    plt.close(fig)


def resolve_input_dir(path: str, fallback_dirs: List[str] | None = None) -> str:
    candidates: List[str] = []
    seen = set()

    def add_candidate(candidate: str) -> None:
        resolved = os.path.abspath(os.path.expanduser(candidate))
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    if os.path.isabs(path):
        add_candidate(path)
    else:
        add_candidate(path)
        add_candidate(os.path.join(PROJECT_ROOT, path))
        add_candidate(os.path.join(SCRIPT_DIR, path))

    for fallback in fallback_dirs or []:
        add_candidate(os.path.join(PROJECT_ROOT, fallback))
        add_candidate(os.path.join(SCRIPT_DIR, fallback))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate input directory '{path}'. Tried: {', '.join(candidates)}"
    )


def load_inputs(stage1_dir: str, stage3_dir: str) -> Dict[str, object]:
    stage1_dir = resolve_input_dir(stage1_dir, fallback_dirs=["outputs/fse/estimation", "fse/estimation"])
    stage3_dir = resolve_input_dir(
        stage3_dir,
        fallback_dirs=["outputs/optimization/solver", "optimization/solver"],
    )

    assignment_detail = {
        budget: pd.read_csv(
            os.path.join(stage3_dir, f"model3_stage3_assignment_lambda0p0_{budget}.csv")
        )
        for budget in BUDGET_ORDER
    }

    return {
        "summary": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_policy_summary_main.csv")),
        "pareto": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_pareto_front_main.csv")),
        "menu": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_frontier_representative_menu.csv")),
        "uvs": pd.read_csv(
            os.path.join(stage3_dir, "model3_stage3_budget_compare_uniform_vs_stratified.csv")
        ),
        "eff": pd.read_csv(
            os.path.join(stage3_dir, "model3_stage3_budget_compare_efficiency_main.csv")
        ),
        "fair": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_budget_compare_fairness.csv")),
        "assign_detail": assignment_detail,
        "mu": pd.read_csv(
            os.path.join(stage1_dir, "model3_stage1_mu_delta_216_balanced_main.csv")
        ),
    }


def budget_label(name: str) -> str:
    mapping = {
        "tight": "11亿元\n紧约束",
        "moderate": "22亿元\n中等预算",
        "ample": "44亿元\n较充裕",
        "max": "89亿元\n最大预算",
    }
    return mapping.get(name, name)


def budget_label_inline(name: str) -> str:
    mapping = {
        "tight": "11亿元（紧约束）",
        "moderate": "22亿元（中等预算）",
        "ample": "44亿元（较充裕）",
        "max": "89亿元（最大预算）",
    }
    return mapping.get(name, name)


def wrap_text(text: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def effect_to_wan(effect_units: float) -> float:
    return effect_units / 10000.0


def short_group_label(group_id: str, mark_vulnerable: bool = False) -> str:
    p, i, g = map(int, group_id.split("-"))
    p_map = {1: "首胎", 2: "二胎", 3: "三孩"}
    i_map = {1: "中低", 2: "中高"}
    g_map = {1: "无祖辈", 2: "有祖辈"}
    label = f"{p_map[p]}|{i_map[i]}|{g_map[g]}"
    if mark_vulnerable and is_vulnerable_group(group_id):
        return f"* {label}"
    return label


def is_vulnerable_group(group_id: str) -> bool:
    p, i, g = map(int, group_id.split("-"))
    return (p >= 2) and (i == 1) and (g == 1)


def group_sort_key(group_id: str) -> Tuple[int, int, int, int]:
    p, i, g = map(int, group_id.split("-"))
    vulnerable = int(is_vulnerable_group(group_id))
    return (-vulnerable, -p, i, g)


def annotate_source(fig: plt.Figure, text: str) -> None:
    fig.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=8.5, color=PALETTE["slate"])


def _bubble_sizes(values: Iterable[float], min_size: float = 280, max_size: float = 1600) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if np.allclose(arr.max(), arr.min()):
        return np.full_like(arr, (min_size + max_size) / 2.0)
    scaled = (arr - arr.min()) / (arr.max() - arr.min())
    return min_size + scaled * (max_size - min_size)


def plot_policy_space_matrix(summary: pd.DataFrame, outdir: Dict[str, str]) -> None:
    df = summary.copy()
    df["cost_yi"] = df["total_cost_yuan"] / 1e8
    df["effect_wan"] = df["total_effect_units"].map(effect_to_wan)

    norm = TwoSlopeNorm(vmin=df["effect_wan"].min(), vcenter=0.0, vmax=df["effect_wan"].max())
    sizes = _bubble_sizes(df["cost_yi"])
    size_map = dict(zip(df["policy_id"], sizes))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), sharey=True)
    cmap = LinearSegmentedColormap.from_list(
        "effect_space",
        ["#E8EEF2", "#A7C0CB", PALETTE["gold"], PALETTE["red"]],
    )

    for ax, childcare_level in zip(axes, [1, 2]):
        sub = df[df["C"] == childcare_level].copy().sort_values(["L", "S"])
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0.5, 3.5)
        ax.set_xticks([1, 2, 3])
        ax.set_yticks([1, 2, 3])
        ax.set_xticklabels(["低补贴 S=1", "中补贴 S=2", "高补贴 S=3"])
        ax.set_yticklabels(["短假期 L=1", "中假期 L=2", "长假期 L=3"])
        ax.set_facecolor(PALETTE["sand"])
        ax.grid(True, color=PALETTE["white"], linewidth=2.0)
        ax.set_title(f"{'基础托育' if childcare_level == 1 else '增强托育'}（C={childcare_level}）", pad=12)
        ax.set_xlabel("补贴强度")

        ax.scatter(
            sub["S"],
            sub["L"],
            s=[size_map[x] for x in sub["policy_id"]],
            c=sub["effect_wan"],
            cmap=cmap,
            norm=norm,
            edgecolors=[
                PALETTE["navy"] if flag else PALETTE["white"]
                for flag in sub["is_pareto_front"]
            ],
            linewidths=[2.6 if flag else 1.3 for flag in sub["is_pareto_front"]],
            zorder=3,
        )

        for _, row in sub.iterrows():
            ax.text(
                row["S"],
                row["L"] + 0.08,
                row["policy_id"],
                ha="center",
                va="bottom",
                fontsize=10.2,
                color=PALETTE["white"] if row["effect_wan"] > 18 else PALETTE["ink"],
                fontweight="bold",
                zorder=4,
            )
            ax.text(
                row["S"],
                row["L"] - 0.10,
                f"{row['effect_wan']:.1f}万\n{row['cost_yi']:.1f}亿",
                ha="center",
                va="top",
                fontsize=8.6,
                color=PALETTE["white"] if row["effect_wan"] > 18 else PALETTE["ink"],
                zorder=4,
            )

    axes[0].set_ylabel("假期强度")

    legend_costs = [25, 90, 180]
    legend_handles = [
        plt.scatter([], [], s=_bubble_sizes([v], 280, 1600)[0], color=PALETTE["grey"], edgecolors="none")
        for v in legend_costs
    ]
    front_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=PALETTE["light_grey"],
        markeredgecolor=PALETTE["navy"],
        markeredgewidth=2.2,
        markersize=10,
        label="黑框 = 帕累托前沿点",
    )

    fig.legend(
        legend_handles,
        [f"{v}亿元" for v in legend_costs],
        title="圆点大小 = 总成本",
        loc="lower center",
        bbox_to_anchor=(0.42, 0.025),
        ncol=3,
        columnspacing=1.6,
        handletextpad=1.0,
    )
    fig.legend(
        handles=[front_handle],
        loc="lower center",
        bbox_to_anchor=(0.78, 0.025),
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.028, pad=0.06)
    cbar.set_label("总效果（万单位）")

    fig.suptitle("图1  18个政策包的离散策略空间", x=0.02, y=0.98, ha="left", fontsize=18, color=PALETTE["ink"])
    fig.text(
        0.02,
        0.93,
        "颜色表示总效果，圆点大小表示总成本，黑框点构成帕累托前沿。该问题更接近离散策略矩阵，而不是连续散点云。",
        ha="left",
        va="top",
        fontsize=10.5,
        color=PALETTE["slate"],
    )
    annotate_source(fig, "Source: model3_stage3_policy_summary_main.csv")
    fig.subplots_adjust(left=0.08, right=0.84, top=0.84, bottom=0.17, wspace=0.06)
    save_figure(fig, outdir["figures"], "fig01_policy_space_matrix")


def plot_frontier_and_budget_gate(
    summary: pd.DataFrame,
    menu: pd.DataFrame,
    uvs: pd.DataFrame,
    outdir: Dict[str, str],
) -> None:
    df = summary.copy()
    df["cost_yi"] = df["total_cost_yuan"] / 1e8
    df["effect_wan"] = df["total_effect_units"].map(effect_to_wan)
    front = df[df["is_pareto_front"]].copy().sort_values("cost_yi")
    front["marginal_effect_per_yi"] = front["total_effect_units"].diff() / front["cost_yi"].diff()

    menu_ids = set(menu["policy_id"])
    all_policies = df.sort_values("cost_yi")

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.5), gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax = axes[0]
    ax.scatter(
        all_policies["cost_yi"],
        all_policies["effect_wan"],
        s=42,
        color=PALETTE["light_grey"],
        zorder=1,
    )
    ax.plot(front["cost_yi"], front["effect_wan"], color=PALETTE["red"], linewidth=2.8, zorder=3)
    ax.scatter(front["cost_yi"], front["effect_wan"], s=92, color=PALETTE["white"], zorder=4)
    ax.scatter(
        front["cost_yi"],
        front["effect_wan"],
        s=62,
        color=PALETTE["red"],
        edgecolor=PALETTE["white"],
        linewidth=1.2,
        zorder=5,
    )

    label_offsets = {
        "1-1-1": (10, -18),
        "1-1-2": (10, 14),
        "1-2-1": (10, 14),
        "1-3-1": (10, 14),
        "3-1-2": (10, 12),
        "3-2-1": (10, 14),
        "3-3-1": (10, 12),
        "3-3-2": (10, 12),
    }
    for _, row in front.iterrows():
        dx, dy = label_offsets.get(row["policy_id"], (8, 8))
        color = PALETTE["navy"] if row["policy_id"] in menu_ids else PALETTE["ink"]
        weight = "bold" if row["policy_id"] in menu_ids else "normal"
        ax.annotate(
            row["policy_id"],
            xy=(row["cost_yi"], row["effect_wan"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.2,
            color=color,
            fontweight=weight,
        )

    for idx in range(1, len(front)):
        prev = front.iloc[idx - 1]
        curr = front.iloc[idx]
        mid_x = (prev["cost_yi"] + curr["cost_yi"]) / 2
        mid_y = (prev["effect_wan"] + curr["effect_wan"]) / 2
        marginal = curr["marginal_effect_per_yi"]
        if pd.notna(marginal):
            ax.text(
                mid_x,
                mid_y + 1.6,
                f"+{marginal:,.0f}/亿元",
                fontsize=8.8,
                color=PALETTE["slate"],
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=PALETTE["grid"], lw=0.8),
            )

    ax.set_title("A. 帕累托前沿的阶梯收益", loc="left", pad=10)
    ax.set_xlabel("总成本（亿元）")
    ax.set_ylabel("总效果（万单位）")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    uniform = uvs[(uvs["lambda_fair"] == 0.0) & (uvs["mode"] == "uniform")].copy()
    uniform["budget_name"] = pd.Categorical(uniform["budget_name"], categories=BUDGET_ORDER, ordered=True)
    uniform = uniform.sort_values("budget_name")
    uniform["budget_yi"] = uniform["budget_name"].map(BUDGET_YI)
    uniform["actual_cost_yi"] = uniform["total_cost_yi_yuan"]
    min_positive_cost = (
        summary.loc[summary["total_cost_yuan"] > 0, "total_cost_yuan"].min() / 1e8
    )

    x = np.arange(len(uniform))
    ax.bar(
        x,
        uniform["budget_yi"],
        color=PALETTE["sand"],
        edgecolor=PALETTE["grey"],
        linewidth=1.5,
        width=0.64,
        zorder=1,
        label="预算上限",
    )
    ax.bar(
        x,
        uniform["actual_cost_yi"],
        color=PALETTE["teal"],
        width=0.64,
        zorder=2,
        label="统一投放实际使用",
    )
    ax.axhline(min_positive_cost, color=PALETTE["red"], linewidth=2.0, linestyle="--")
    ax.text(
        0.05,
        min_positive_cost + 2.0,
        f"最便宜非基准包门槛：{min_positive_cost:.1f}亿元",
        color=PALETTE["red"],
        fontsize=10,
    )

    for idx, row in uniform.reset_index(drop=True).iterrows():
        if row["actual_cost_yi"] == 0:
            label = "只能选基准\n1-1-1"
        else:
            label = f"{row['selected_policy_id']}\n{effect_to_wan(row['total_effect_units']):.1f}万效果"
        ax.text(
            idx,
            row["budget_yi"] + 2.8,
            label,
            ha="center",
            va="bottom",
            fontsize=9.6,
            color=PALETTE["ink"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([budget_label(x) for x in uniform["budget_name"]])
    ax.set_ylabel("预算 / 实际支出（亿元）")
    ax.set_title("B. 统一投放的预算门槛", loc="left", pad=10)
    ax.set_ylim(0, max(uniform["budget_yi"]) * 1.18)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    fig.suptitle("图2  前沿选择与统一投放门槛", x=0.02, y=0.98, ha="left", fontsize=18)
    fig.text(
        0.02,
        0.93,
        "左图展示前沿点并标出每增加1亿元带来的边际效果；右图展示统一投放为何在11/22亿元预算下无法跨过最小可行门槛。",
        ha="left",
        va="top",
        fontsize=10.5,
        color=PALETTE["slate"],
    )
    annotate_source(
        fig,
        "Source: model3_stage3_policy_summary_main.csv, model3_stage3_frontier_representative_menu.csv, model3_stage3_budget_compare_uniform_vs_stratified.csv",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save_figure(fig, outdir["figures"], "fig02_frontier_and_budget_gate")


def plot_uniform_vs_stratified_gain(uvs: pd.DataFrame, outdir: Dict[str, str]) -> None:
    df = uvs[(uvs["lambda_fair"] == 0.0) & (uvs["mode"].isin(["uniform", "stratified"]))].copy()
    df["budget_name"] = pd.Categorical(df["budget_name"], categories=BUDGET_ORDER, ordered=True)
    df = df.sort_values(["budget_name", "mode"])
    df["effect_wan"] = df["total_effect_units"].map(effect_to_wan)

    pivot = (
        df.pivot_table(index="budget_name", columns="mode", values="effect_wan", aggfunc="first")
        .reindex(BUDGET_ORDER)
    )
    meta = (
        df[df["mode"] == "stratified"][
            ["budget_name", "gain_vs_uniform_effect_pct", "n_unique_policies", "effect_wan"]
        ]
        .set_index("budget_name")
        .reindex(BUDGET_ORDER)
    )

    diff = pivot["stratified"] - pivot["uniform"]

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), gridspec_kw={"width_ratios": [1.15, 0.95]})

    ax = axes[0]
    x = np.arange(len(BUDGET_ORDER))
    width = 0.34
    ax.bar(x - width / 2, pivot["uniform"], width=width, color=PALETTE["grey"], label="统一投放")
    ax.bar(x + width / 2, pivot["stratified"], width=width, color=PALETTE["red"], label="分层投放")

    for idx, budget in enumerate(BUDGET_ORDER):
        ax.text(
            idx - width / 2,
            pivot.loc[budget, "uniform"] + 0.9,
            f"{pivot.loc[budget, 'uniform']:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.4,
            color=PALETTE["ink"],
        )
        ax.text(
            idx + width / 2,
            pivot.loc[budget, "stratified"] + 0.9,
            f"{pivot.loc[budget, 'stratified']:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.4,
            color=PALETTE["ink"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([budget_label_inline(x) for x in BUDGET_ORDER], fontsize=10.5)
    ax.set_ylabel("总效果（万单位）")
    ax.set_title("A. 绝对效果", loc="left", pad=10)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    ax = axes[1]
    colors = [PALETTE["rose"] if budget in ("tight", "moderate") else PALETTE["red"] for budget in BUDGET_ORDER]
    bars = ax.barh(np.arange(len(BUDGET_ORDER)), diff[BUDGET_ORDER], color=colors, height=0.58)

    for idx, budget in enumerate(BUDGET_ORDER):
        pct = meta.loc[budget, "gain_vs_uniform_effect_pct"]
        note = f"+{pct:.1f}%" if pd.notna(pct) else "从0起步"
        ax.text(
            bars[idx].get_width() + max(diff) * 0.03,
            idx,
            f"{note}  |  n={int(meta.loc[budget, 'n_unique_policies'])}",
            va="center",
            fontsize=10.1,
            color=PALETTE["ink"],
        )

    ax.set_yticks(np.arange(len(BUDGET_ORDER)))
    ax.set_yticklabels([budget_label_inline(x) for x in BUDGET_ORDER], fontsize=10.5)
    ax.set_xlabel("分层投放相对统一投放的新增效果（万单位）")
    ax.set_title("B. 分层投放的增量收益", loc="left", pad=10)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("图3  分层投放如何把预算转成更高效果", x=0.02, y=0.98, ha="left", fontsize=18)
    fig.text(
        0.02,
        0.93,
        "低预算阶段，分层投放解决了统一投放“无法起步”的问题；44亿元与89亿元阶段，分层带来的效果提升仍然非常显著。",
        ha="left",
        va="top",
        fontsize=10.5,
        color=PALETTE["slate"],
    )
    annotate_source(fig, "Source: model3_stage3_budget_compare_uniform_vs_stratified.csv")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save_figure(fig, outdir["figures"], "fig03_uniform_vs_stratified_gain")


def plot_allocation_concentration(assign_detail: Dict[str, pd.DataFrame], outdir: Dict[str, str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.8), sharex=False)
    axes = axes.flatten()

    max_share = 0.0
    prepared: Dict[str, pd.DataFrame] = {}
    for budget in BUDGET_ORDER:
        df = assign_detail[budget].copy()
        df["budget_share_pct"] = df["group_total_cost_yuan"] / df["group_total_cost_yuan"].sum() * 100
        df["effect_share_pct"] = df["group_total_effect_units"] / df["group_total_effect_units"].sum() * 100
        df["group_short"] = df["group_id"].map(lambda x: short_group_label(x, mark_vulnerable=True))
        df = df.sort_values("budget_share_pct", ascending=False).reset_index(drop=True)
        prepared[budget] = df
        max_share = max(max_share, df["budget_share_pct"].max())

    for ax, budget in zip(axes, BUDGET_ORDER):
        df = prepared[budget]
        y = np.arange(len(df))
        ax.barh(y, df["budget_share_pct"], color=PALETTE["mist"], height=0.62, label="预算占比")
        ax.scatter(df["effect_share_pct"], y, color=PALETTE["red"], s=54, zorder=3, label="效果占比")

        for idx, row in df.iterrows():
            ax.text(
                max(row["budget_share_pct"], row["effect_share_pct"]) + max_share * 0.03,
                idx,
                row["policy_id"],
                va="center",
                fontsize=8.8,
                color=PALETTE["ink"],
            )

        ax.set_yticks(y)
        ax.set_yticklabels(df["group_short"], fontsize=8.8)
        ax.invert_yaxis()
        ax.set_title(budget_label_inline(budget), fontsize=12.5, loc="left", pad=8)
        ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, max_share * 1.34)
        if budget in ("ample", "max"):
            ax.set_xlabel("占比（%）")

    handles = [
        Line2D([0], [0], color=PALETTE["mist"], linewidth=9, label="预算占比"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["red"], markeredgecolor="none", markersize=8, label="效果占比"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.93), ncol=2)

    fig.suptitle("图4  分层投放的资源集中度与组别配置", x=0.02, y=0.98, ha="left", fontsize=18)
    fig.text(
        0.02,
        0.93,
        "每个子图按预算占比从高到低排序；浅条是预算占比，红点是效果占比，右侧文字是该家庭组拿到的政策包。带 * 的组为脆弱组。",
        ha="left",
        va="top",
        fontsize=10.3,
        color=PALETTE["slate"],
    )
    annotate_source(
        fig,
        "Source: model3_stage3_assignment_lambda0p0_tight/moderate/ample/max.csv",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    save_figure(fig, outdir["figures"], "fig04_allocation_concentration")


def plot_fairness_arrow_strip(eff: pd.DataFrame, fair: pd.DataFrame, outdir: Dict[str, str]) -> None:
    base = eff[(eff["mode"] == "stratified") & (eff["lambda_fair"] == 0.0)].copy()
    other = fair[(fair["mode"] == "stratified") & (fair["lambda_fair"].isin([0.5, 1.0]))].copy()
    df = pd.concat([base, other], ignore_index=True)

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.8))
    color_map = {0.0: PALETTE["teal"], 0.5: PALETTE["gold"], 1.0: PALETTE["red"]}
    marker_map = {0.0: "o", 0.5: "s", 1.0: "^"}

    for ax, budget in zip(axes, BUDGET_ORDER):
        sub = df[df["budget_name"] == budget].copy()
        sub["lambda_fair"] = pd.Categorical(sub["lambda_fair"], categories=[0.0, 0.5, 1.0], ordered=True)
        sub = sub.sort_values("lambda_fair").reset_index(drop=True)

        base_total = sub.loc[sub["lambda_fair"] == 0.0, "total_effect_units"].iloc[0]
        base_vul = sub.loc[sub["lambda_fair"] == 0.0, "vulnerable_effect_units"].iloc[0]
        sub["delta_total"] = sub["total_effect_units"] - base_total
        sub["delta_vul"] = sub["vulnerable_effect_units"] - base_vul

        ax.axhline(0, color=PALETTE["light_grey"], linewidth=1.0, zorder=0)
        ax.axvline(0, color=PALETTE["light_grey"], linewidth=1.0, zorder=0)

        points = sub[["delta_vul", "delta_total"]].to_numpy()
        for idx in range(len(points) - 1):
            ax.annotate(
                "",
                xy=points[idx + 1],
                xytext=points[idx],
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=PALETTE["slate"]),
            )

        label_groups: Dict[Tuple[float, float], List[str]] = {}
        for _, row in sub.iterrows():
            lam = float(row["lambda_fair"])
            ax.scatter(
                row["delta_vul"],
                row["delta_total"],
                s=92 if lam > 0 else 82,
                marker=marker_map[lam],
                color=color_map[lam],
                edgecolor=PALETTE["white"],
                linewidth=1.2,
                zorder=3,
            )
            label_key = (round(float(row["delta_vul"]), 6), round(float(row["delta_total"]), 6))
            label_text = "λ=0" if lam == 0 else f"λ={lam:.1f}"
            label_groups.setdefault(label_key, []).append(label_text)

        for (xv, yv), labels in label_groups.items():
            combined = "/".join(labels)
            dy = -14 if len(labels) == 1 and "λ=1.0" in labels[0] else 8
            ax.annotate(
                combined,
                xy=(xv, yv),
                xytext=(8, dy),
                textcoords="offset points",
                fontsize=9.2,
                color=PALETTE["ink"],
            )

        max_x = max(abs(sub["delta_vul"]).max(), 1.0)
        max_y = max(abs(sub["delta_total"]).max(), 1.0)
        ax.set_xlim(-0.14 * max_x, max_x * 1.18)
        ax.set_ylim(min(sub["delta_total"].min() * 1.18, -0.14 * max_y), max(sub["delta_total"].max() * 1.18, 0.14 * max_y))
        ax.set_title(budget_label_inline(budget), fontsize=11.8, pad=8)
        ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if budget == "tight":
            ax.set_ylabel("总效果变化（相对 λ=0）")
        ax.set_xlabel("脆弱组效果变化（相对 λ=0）")

    fig.suptitle("图5  公平加权如何改变脆弱组与总效果之间的取舍", x=0.02, y=1.02, ha="left", fontsize=18)
    fig.text(
        0.02,
        0.955,
        "每个预算档位展示 λ=0 → 0.5 → 1.0 的移动方向。箭头越向右，说明脆弱组收益越高；越向下，说明总效果牺牲越大。",
        ha="left",
        va="top",
        fontsize=10.3,
        color=PALETTE["slate"],
    )
    annotate_source(fig, "Source: model3_stage3_budget_compare_efficiency_main.csv, model3_stage3_budget_compare_fairness.csv")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    save_figure(fig, outdir["figures"], "fig05_fairness_arrow_strip")


def plot_group_policy_heterogeneity(mu: pd.DataFrame, outdir: Dict[str, str]) -> None:
    key_policies = ["1-2-1", "1-3-1", "3-1-2", "3-2-1", "3-3-1", "3-3-2"]
    df = mu[mu["policy_id"].isin(key_policies)].copy()
    group_order = sorted(df["group_id"].unique(), key=group_sort_key)
    df["group_short"] = df["group_id"].map(lambda x: short_group_label(x, mark_vulnerable=True))
    pivot = (
        df.pivot_table(index="group_short", columns="policy_id", values="delta_hat_zj", aggfunc="first")
        .reindex([short_group_label(g, mark_vulnerable=True) for g in group_order])
        [key_policies]
    )

    vmax = np.nanmax(np.abs(pivot.values))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("heterogeneity", ["#C8D8E0", "#F9F4EC", "#D85C4A"])

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(key_policies)))
    ax.set_xticklabels(key_policies, fontsize=10.5)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9.2)
    ax.set_title("附图A1  家庭组别对关键政策包的异质性响应", loc="left", pad=10)
    ax.set_xlabel("关键政策包")
    ax.set_ylabel("家庭组别（* 为脆弱组）")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color=PALETTE["white"] if abs(value) > vmax * 0.55 else PALETTE["ink"],
            )

    ax.set_xticks(np.arange(-0.5, len(key_policies), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("delta_hat_zj")
    annotate_source(fig, "Source: model3_stage1_mu_delta_216_balanced_main.csv")
    fig.tight_layout()
    save_figure(fig, outdir["appendix"], "figA1_group_policy_heterogeneity")


def write_canva_layout_guide(outdir: Dict[str, str]) -> None:
    guide = """# Canva Layout Guide

This folder contains presentation-ready chart assets exported from Python.

## Suggested Slide Flow

1. Cover
   - Title: 浙江口径下生育支持政策的成本-效果与分层投放优化
   - Subtitle: 统一投放存在预算门槛，分层投放在有限预算下显著提升总效果

2. Research Setup
   - Show a simple Canva-made process strip:
     Stage 1 效果估计 -> Stage 2 成本映射 -> Stage 3 预算优化

3. Policy Space
   - Use: `figures/fig01_policy_space_matrix.svg`
   - Add a small legend card explaining S / L / C

4. Frontier And Budget Gate
   - Use: `figures/fig02_frontier_and_budget_gate.svg`
   - Call out the minimum non-baseline threshold: about 25.2 亿元

5. Stratified Advantage
   - Use: `figures/fig03_uniform_vs_stratified_gain.svg`
   - Add large number cards for +229.3% and +70.5%

6. Allocation Concentration
   - Use: `figures/fig04_allocation_concentration.svg`
   - Highlight groups `2-1-2` and `2-1-1`

7. Fairness Trade-off
   - Use: `figures/fig05_fairness_arrow_strip.svg`
   - Add one sentence per budget about the fairness-efficiency trade-off

8. Appendix
   - Use: `appendix/figA1_group_policy_heterogeneity.svg`
   - Useful when explaining why stratification dominates uniform allocation

## Canva Notes

- Prefer SVG imports for editing flexibility.
- Keep page titles and takeaway text in Canva, not in Python annotations.
- Use one accent color for conclusions and one muted color for supporting detail.
- Limit each page to one chart plus 2-3 takeaway bullets.
"""
    with open(os.path.join(outdir["root"], "CANVA_LAYOUT_GUIDE.md"), "w", encoding="utf-8") as fh:
        fh.write(guide)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "fse", "estimation"),
        help="FSE estimation results directory",
    )
    parser.add_argument(
        "--stage3-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "optimization", "solver"),
        help="Optimization solver results directory",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "optimization", "visualization_v3"),
        help="Output directory for v3 presentation assets",
    )
    args = parser.parse_args()

    set_global_style()
    outdir = ensure_dirs(args.output_dir)
    dfs = load_inputs(args.stage1_dir, args.stage3_dir)

    plot_policy_space_matrix(dfs["summary"], outdir)
    plot_frontier_and_budget_gate(dfs["summary"], dfs["menu"], dfs["uvs"], outdir)
    plot_uniform_vs_stratified_gain(dfs["uvs"], outdir)
    plot_allocation_concentration(dfs["assign_detail"], outdir)
    plot_fairness_arrow_strip(dfs["eff"], dfs["fair"], outdir)
    plot_group_policy_heterogeneity(dfs["mu"], outdir)
    write_canva_layout_guide(outdir)


if __name__ == "__main__":
    main()
