#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model 3 visualization suite (v2)
--------------------------------
Generate:
    Table 1  政策映射表
    Figure 1 成本—效果帕累托前沿主图
    Table 2  代表性预算菜单
    Figure 2 统一投放 vs 分层投放哑铃图
    Figure 3 分层投放三联热力图
    Figure 4 公平—效率影响平面（可选，v2 为相对变化平面）

Default data inputs:
    /mnt/data/model3_stage2
    /mnt/data/model3_stage3

Outputs:
    <output_dir>/tables/*.png
    <output_dir>/figures/*.png
    <output_dir>/tables/*.csv
    <output_dir>/tables/*.tex
"""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


PALETTE = {
    "red": "#D93F49",
    "rose": "#E28187",
    "pink": "#EBBFC2",
    "mist": "#D5E1E3",
    "bluegrey": "#AFC9CF",
    "teal": "#8FB4BE",
    "ink": "#4A5560",
    "grid": "#E7ECEE",
}


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
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["axes.labelsize"] = 13
    matplotlib.rcParams["legend.frameon"] = False
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.dpi"] = 300


def ensure_dirs(output_dir: str) -> Dict[str, str]:
    out = {
        "root": output_dir,
        "figures": os.path.join(output_dir, "figures"),
        "tables": os.path.join(output_dir, "tables"),
    }
    for p in out.values():
        os.makedirs(p, exist_ok=True)
    return out


def budget_label(name: str) -> str:
    mapping = {
        "tight": "11亿元\n紧约束",
        "moderate": "22亿元\n中等预算",
        "ample": "44亿元\n较充裕预算",
        "max": "89亿元\n最大预算",
    }
    return mapping.get(name, name)


def budget_label_inline(name: str) -> str:
    mapping = {
        "tight": "11亿元（紧约束）",
        "moderate": "22亿元（中等预算）",
        "ample": "44亿元（较充裕预算）",
        "max": "89亿元（最大预算）",
    }
    return mapping.get(name, name)


def wrap_text(s: str, width: int = 14) -> str:
    if pd.isna(s):
        return ""
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def savefig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight")
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


def yuan_to_yi(yuan: float) -> float:
    return yuan / 1e8


def _policy_meaning(policy_id: str) -> str:
    s, l, c = map(int, policy_id.split("-"))
    if (s, l, c) == (1, 1, 2):
        return "仅增强托育，试探性低成本托底"
    if s == 1 and l >= 2 and c == 1:
        return "以假期增强为主的低成本制度微调"
    if s == 3 and l == 3 and c == 2:
        return "三工具同步高强度加码，追求最大效果"
    if s == 3 and l >= 2 and c == 1:
        return "补贴+假期双加码，兼顾效果与预算"
    if c == 2 and s >= 2 and l >= 2:
        return "在补贴与假期基础上叠加托育强化"
    if s >= 2 and c == 2:
        return "现金补贴配合托育减负的折中方案"
    if l >= 2 and c == 2:
        return "假期与托育协同，缓解照护约束"
    return "组合式增量支持方案"


def _criterion_to_budget_view(criterion: str) -> str:
    mapping = {
        "最低成本前沿点": "低成本起步",
        "35%分位附近前沿点": "中等预算折中",
        "72%分位附近前沿点": "较充裕预算",
        "最高效果前沿点": "最大效果",
    }
    return mapping.get(criterion, criterion)


def _compact_group_label(group_id: str) -> str:
    p, i, g = map(int, group_id.split("-"))
    p_map = {1: "首胎", 2: "二孩", 3: "三孩"}
    i_map = {1: "中低收入", 2: "中高收入"}
    g_map = {1: "无祖辈", 2: "有祖辈"}
    return f"{p_map[p]}｜{i_map[i]}｜{g_map[g]}"


def _group_sort_key(group_id: str) -> Tuple[int, int, int, int]:
    p, i, g = map(int, group_id.split("-"))
    vulnerable = int((p >= 2) and (i == 1) and (g == 1))
    return (-vulnerable, -p, g, i)


def _format_numeric_for_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if "亿元" in col:
                out[col] = out[col].map(lambda x: f"{x:.2f}")
            elif "元/户" in col:
                out[col] = out[col].map(lambda x: f"{x:,.0f}")
            elif "每1万元" in col:
                out[col] = out[col].map(lambda x: f"{x:.3f}")
            elif "效果" in col:
                out[col] = out[col].map(lambda x: f"{x:.3f}")
            else:
                out[col] = out[col].map(lambda x: f"{x}")
    return out


def render_table_png(
    df: pd.DataFrame,
    path: str,
    title: str,
    col_widths: List[float] | None = None,
    font_size: int = 10,
    row_height_scale: float = 1.5,
    zebra: bool = True,
) -> None:
    df_show = df.copy()
    for col in df_show.columns:
        if df_show[col].dtype == object:
            width = 10 if len(df_show) > 6 else 14
            df_show[col] = df_show[col].map(lambda x: wrap_text(x, width=width))
    nrows, ncols = df_show.shape
    if col_widths is None:
        col_widths = [1.0 / ncols] * ncols
    fig_w = max(10, 1.8 * ncols + 2)
    fig_h = max(3.2, 0.58 * (nrows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=12, color=PALETTE["ink"])

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_height_scale)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if r == 0:
            cell.set_facecolor(PALETTE["teal"])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F8FAFA" if zebra and r % 2 == 0 else "white")
            cell.set_text_props(color=PALETTE["ink"])

    savefig(fig, path)


def load_inputs(stage2_dir: str, stage3_dir: str) -> Dict[str, pd.DataFrame]:
    stage2_dir = resolve_input_dir(stage2_dir, fallback_dirs=["stage2_cost_mapping"])
    stage3_dir = resolve_input_dir(
        stage3_dir,
        fallback_dirs=["stage3_optmization", "stage3_optimization"],
    )

    return {
        "mapping": pd.read_csv(os.path.join(stage2_dir, "model3_stage2_policy_mapping_table.csv")),
        "summary": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_policy_summary_main.csv")),
        "pareto": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_pareto_front_main.csv")),
        "menu": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_frontier_representative_menu.csv")),
        "uvs": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_budget_compare_uniform_vs_stratified.csv")),
        "eff": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_budget_compare_efficiency_main.csv")),
        "fair": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_budget_compare_fairness.csv")),
        "assign0": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_assignment_matrix_lambda0p0.csv")),
        "assign1": pd.read_csv(os.path.join(stage3_dir, "model3_stage3_assignment_matrix_lambda1p0.csv")),
    }


def build_policy_mapping_table(mapping: pd.DataFrame) -> pd.DataFrame:
    out = mapping.copy()
    dim_map = {"S": "现金补贴", "L": "假期支持", "C": "托育支持"}
    out["dimension"] = out["dimension"].map(dim_map).fillna(out["dimension"])
    out = out.rename(
        columns={
            "dimension": "政策维度",
            "level": "水平",
            "label": "实验水平",
            "zhejiang_mapping": "浙江口径映射",
            "notes": "说明",
        }
    )
    return out[["政策维度", "水平", "实验水平", "浙江口径映射", "说明"]]


def export_table1(mapping: pd.DataFrame, outdir: Dict[str, str]) -> pd.DataFrame:
    table1 = build_policy_mapping_table(mapping)
    table1.to_csv(os.path.join(outdir["tables"], "table1_政策映射表.csv"), index=False, encoding="utf-8-sig")
    table1.to_latex(os.path.join(outdir["tables"], "table1_政策映射表.tex"), index=False, escape=False)
    render_table_png(
        table1,
        os.path.join(outdir["tables"], "table1_政策映射表.png"),
        title="表1  实验政策属性到浙江公共支出的映射规则",
        col_widths=[0.13, 0.08, 0.16, 0.40, 0.23],
        font_size=10,
        row_height_scale=1.7,
    )
    return table1


def _draw_round_rect_label(ax, x, y, text, dx, dy, align="left", width=12):
    ax.annotate(
        wrap_text(text, width=width),
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=align,
        va="center",
        fontsize=9.6,
        color=PALETTE["ink"],
        bbox=dict(
            boxstyle="round,pad=0.28,rounding_size=0.12",
            fc="white",
            ec=PALETTE["grid"],
            lw=1.0,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=PALETTE["bluegrey"],
            lw=1.0,
            shrinkA=0,
            shrinkB=5,
            connectionstyle="angle3,angleA=0,angleB=90",
        ),
        zorder=10,
    )


def plot_pareto_front(summary: pd.DataFrame, menu: pd.DataFrame, outdir: Dict[str, str]) -> None:
    df = summary.copy()
    df["total_cost_yi_yuan"] = df["total_cost_yuan"].map(yuan_to_yi)
    front = df[df["is_pareto_front"]].copy().sort_values("total_cost_yi_yuan")
    nonfront = df[~df["is_pareto_front"]].copy()
    menu_anno = menu.copy()
    menu_anno["total_cost_yi_yuan"] = menu_anno["total_cost_yuan"].map(yuan_to_yi)

    fig, ax = plt.subplots(figsize=(10.8, 6.6))

    # background layer: dominated policies
    ax.scatter(
        nonfront["total_cost_yi_yuan"],
        nonfront["total_effect_units"],
        s=58,
        color=PALETTE["mist"],
        edgecolor="none",
        alpha=0.9,
        zorder=1,
    )

    # pareto frontier
    ax.plot(
        front["total_cost_yi_yuan"],
        front["total_effect_units"],
        color=PALETTE["red"],
        linewidth=2.6,
        zorder=3,
    )
    ax.scatter(
        front["total_cost_yi_yuan"],
        front["total_effect_units"],
        s=88,
        color="white",
        edgecolor="white",
        linewidth=0,
        zorder=4,
    )
    ax.scatter(
        front["total_cost_yi_yuan"],
        front["total_effect_units"],
        s=62,
        color=PALETTE["rose"],
        edgecolor=PALETTE["red"],
        linewidth=1.2,
        zorder=5,
    )

    # representative menu points
    for _, r in menu_anno.iterrows():
        x = r["total_cost_yi_yuan"]
        y = df.loc[df["policy_id"] == r["policy_id"], "total_effect_units"].iloc[0]
        ax.scatter([x], [y], s=190, color="white", edgecolor="white", linewidth=0, zorder=6)
        ax.scatter([x], [y], s=116, color=PALETTE["red"], edgecolor="white", linewidth=1.5, zorder=7)

    # baseline annotation
    base = df.loc[df["policy_id"] == "1-1-1"].iloc[0]
    ax.annotate(
        "基准组\n1-1-1",
        xy=(base["total_cost_yi_yuan"], base["total_effect_units"]),
        xytext=(8, -34),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        va="center",
        color=PALETTE["ink"],
    )

    ax.set_xlabel("年度增量支出（亿元）")
    ax.set_ylabel("总效果（年度 cohort 总评分增益）")

    ax.grid(True, axis="both", color=PALETTE["grid"], linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=11)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))
    ax.set_xlim(-10, max(df["total_cost_yi_yuan"]) * 1.05)
    ax.set_ylim(min(-30000, df["total_effect_units"].min() - 18000), df["total_effect_units"].max() * 1.05)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["mist"], markeredgecolor="none", markersize=8, label="被支配方案"),
        Line2D([0], [0], color=PALETTE["red"], linewidth=2.6, marker="o", markerfacecolor=PALETTE["rose"], markeredgecolor=PALETTE["red"], markersize=8, label="帕累托前沿"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2, handlelength=2.2, columnspacing=1.8)

    savefig(fig, os.path.join(outdir["figures"], "fig1_成本效果帕累托前沿主图.png"))


def build_representative_menu_table(menu: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    add = summary[["policy_id", "E_per_10k_yuan"]].copy()
    out = menu.merge(add, on="policy_id", how="left").copy()
    out["预算视角"] = out["criterion"].map(_criterion_to_budget_view)
    out["户均增量成本（元/户）"] = out["Cbar_zj"]
    out["年度增量支出（亿元）"] = out["total_cost_yi_yuan"]
    out["总效果"] = out["Ebar_zj"]
    out["每1万元对应效果"] = out["E_per_10k_yuan"]
    out["政策含义"] = out["policy_id"].map(_policy_meaning)
    out = out.rename(columns={"policy_id": "政策包", "policy_label": "政策组合"})
    return out[["预算视角", "政策包", "政策组合", "户均增量成本（元/户）", "年度增量支出（亿元）", "总效果", "每1万元对应效果", "政策含义"]]


def export_table2(menu: pd.DataFrame, summary: pd.DataFrame, outdir: Dict[str, str]) -> pd.DataFrame:
    table2 = build_representative_menu_table(menu, summary)
    table2.to_csv(os.path.join(outdir["tables"], "table2_代表性预算菜单.csv"), index=False, encoding="utf-8-sig")
    table2.to_latex(os.path.join(outdir["tables"], "table2_代表性预算菜单.tex"), index=False, escape=False)
    render_table_png(
        _format_numeric_for_table(table2),
        os.path.join(outdir["tables"], "table2_代表性预算菜单.png"),
        title="表2  浙江口径下四档代表性预算菜单",
        col_widths=[0.14, 0.10, 0.16, 0.14, 0.14, 0.08, 0.11, 0.13],
        font_size=9.4,
        row_height_scale=1.65,
    )
    return table2


def plot_uniform_vs_stratified_dumbbell(uvs: pd.DataFrame, outdir: Dict[str, str]) -> None:
    df = uvs.copy()
    df = df[(df["lambda_fair"] == 0.0) & (df["mode"].isin(["uniform", "stratified"]))].copy()
    order = ["tight", "moderate", "ample", "max"]
    df["budget_name"] = pd.Categorical(df["budget_name"], categories=order, ordered=True)
    df = df.sort_values(["budget_name", "mode"])

    pivot = df.pivot_table(index="budget_name", columns="mode", values="total_effect_units", aggfunc="first").reindex(order)
    pct = df[df["mode"] == "stratified"][["budget_name", "gain_vs_uniform_effect_pct"]].set_index("budget_name").reindex(order)

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ypos = np.arange(len(order))[::-1]

    for idx, b in enumerate(order):
        y = ypos[idx]
        xu = pivot.loc[b, "uniform"]
        xs = pivot.loc[b, "stratified"]

        ax.plot([xu, xs], [y, y], color=PALETTE["bluegrey"], linewidth=2.2, zorder=1)
        ax.scatter([xu], [y], s=145, color="white", edgecolor="white", linewidth=0, zorder=2)
        ax.scatter([xu], [y], s=90, color=PALETTE["teal"], edgecolor="white", linewidth=1.3, zorder=3)
        ax.scatter([xs], [y], s=155, color="white", edgecolor="white", linewidth=0, zorder=4)
        ax.scatter([xs], [y], s=100, color=PALETTE["red"], edgecolor="white", linewidth=1.3, zorder=5)

        # direct labels instead of legend
        ax.annotate("统一", xy=(xu, y), xytext=(-10, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=10.2, color=PALETTE["ink"])
        ax.annotate("分层", xy=(xs, y), xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10.2, color=PALETTE["ink"])

        gain = pct.loc[b, "gain_vs_uniform_effect_pct"]
        if pd.notna(gain):
            ax.annotate(
                f"+{gain:.1f}%",
                xy=(xs, y),
                xytext=(48, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=11.2,
                color=PALETTE["red"],
                weight="bold",
            )
        else:
            ax.annotate(
                "统一投放不可升级",
                xy=(xs, y),
                xytext=(48, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10.2,
                color=PALETTE["ink"],
            )

    ax.set_yticks(ypos)
    ax.set_yticklabels([budget_label_inline(x) for x in order], fontsize=12)
    ax.set_xlabel("总效果（年度 cohort 总评分增益）")
    ax.set_title("图2  不同预算下统一投放与分层投放的总效果比较", loc="left", pad=12)

    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.9)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

    x_max = max(pivot.max()) * 1.18
    ax.set_xlim(-0.03 * x_max, x_max)
    savefig(fig, os.path.join(outdir["figures"], "fig2_统一投放vs分层投放哑铃图.png"))


def _policy_matrix_to_level_matrix(assign: pd.DataFrame, level: str) -> pd.DataFrame:
    out = assign.copy()
    idx = {"S": 0, "L": 1, "C": 2}[level]
    for col in ["tight", "moderate", "ample", "max"]:
        out[col] = out[col].astype(str).str.split("-").str[idx].astype(int)
    return out


def plot_triptych_assignment_heatmap(assign0: pd.DataFrame, outdir: Dict[str, str]) -> None:
    df = assign0.copy()
    df["sort_key"] = df["group_id"].map(_group_sort_key)
    df = df.sort_values("sort_key").drop(columns="sort_key")
    df["group_short"] = df["group_id"].map(_compact_group_label)

    mats = {}
    for level in ["S", "L", "C"]:
        tmp = _policy_matrix_to_level_matrix(df[["group_id", "group_short", "tight", "moderate", "ample", "max"]], level)
        mats[level] = tmp.set_index("group_short")[["tight", "moderate", "ample", "max"]]

    cmaps = {
        "S": LinearSegmentedColormap.from_list("smap", ["#FAF0F1", PALETTE["pink"], PALETTE["rose"], PALETTE["red"]]),
        "L": LinearSegmentedColormap.from_list("lmap", ["#F7FAFA", PALETTE["mist"], PALETTE["bluegrey"], PALETTE["teal"]]),
        "C": LinearSegmentedColormap.from_list("cmap", ["#F7FAFA", PALETTE["mist"], PALETTE["teal"]]),
    }
    titles = {"S": "补贴强度 S", "L": "假期强度 L", "C": "托育强度 C"}
    vlims = {"S": (1, 3), "L": (1, 3), "C": (1, 2)}

    fig, axes = plt.subplots(
        1, 3, figsize=(14.2, 8.4), sharey=True,
        gridspec_kw={"wspace": 0.06}
    )
    budget_cols = ["tight", "moderate", "ample", "max"]
    xlabels = [budget_label(x) for x in budget_cols]

    # row separators by major block
    row_labels = mats["S"].index.tolist()
    separators = []
    for i in range(1, len(row_labels)):
        prev = row_labels[i - 1].split("｜")
        curr = row_labels[i].split("｜")
        if prev[0] != curr[0] or prev[2] != curr[2]:
            separators.append(i - 0.5)

    for ax, level in zip(axes, ["S", "L", "C"]):
        mat = mats[level]
        ax.imshow(mat.values, aspect="auto", cmap=cmaps[level], vmin=vlims[level][0], vmax=vlims[level][1])

        ax.set_xticks(np.arange(len(budget_cols)))
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index.tolist(), fontsize=9.5)
        ax.set_title(titles[level], color=PALETTE["ink"], pad=10, fontsize=14, fontweight="bold")

        ax.set_xticks(np.arange(-0.5, len(budget_cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for sep in separators:
            ax.hlines(sep, -0.5, len(budget_cols) - 0.5, color="white", linewidth=3.0, zorder=4)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # only show level values, not full policy codes
        threshold = 2 if level != "C" else 2
        for i in range(mat.shape[0]):
            for j, col in enumerate(budget_cols):
                level_value = int(mat.iloc[i, j])
                ax.text(
                    j, i, f"{level_value}",
                    ha="center", va="center",
                    fontsize=11.5,
                    fontweight="bold",
                    color="white" if level_value >= threshold else PALETTE["ink"],
                )

    axes[0].set_ylabel("家庭处境组（按脆弱性排序）", fontsize=12.5)
    fig.suptitle(
        "图3  不同预算档下各家庭处境的最优政策分配",
        x=0.01, y=0.99, ha="left",
        fontsize=16, fontweight="bold", color=PALETTE["ink"]
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    savefig(fig, os.path.join(outdir["figures"], "fig3_分层投放三联热力图.png"))


def plot_fairness_efficiency_plane(eff: pd.DataFrame, fair: pd.DataFrame, outdir: Dict[str, str]) -> None:
    df0 = eff[(eff["mode"] == "stratified") & (eff["lambda_fair"] == 0.0)].copy()
    df1 = fair[(fair["mode"] == "stratified") & (fair["lambda_fair"].isin([0.5, 1.0]))].copy()
    df = pd.concat([df0, df1], ignore_index=True)

    order = ["tight", "moderate", "ample", "max"]
    lam_order = [0.0, 0.5, 1.0]
    color_map = {0.0: PALETTE["teal"], 0.5: PALETTE["rose"], 1.0: PALETTE["red"]}

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2))
    axes = axes.flatten()

    marker_map = {0.0: "o", 0.5: "s", 1.0: "^"}
    label_offset_map = {0.0: (10, 8), 0.5: (10, 10), 1.0: (10, -2)}

    for ax, b in zip(axes, order):
        sub = df[df["budget_name"] == b].copy()
        sub["lambda_fair"] = pd.Categorical(sub["lambda_fair"], categories=lam_order, ordered=True)
        sub = sub.sort_values("lambda_fair").reset_index(drop=True)

        base_total = sub.loc[sub["lambda_fair"] == 0.0, "total_effect_units"].iloc[0]
        base_vul = sub.loc[sub["lambda_fair"] == 0.0, "vulnerable_effect_units"].iloc[0]
        sub["delta_total"] = sub["total_effect_units"] - base_total
        sub["delta_vul"] = sub["vulnerable_effect_units"] - base_vul

        ax.axhline(0, color=PALETTE["grid"], linewidth=1.1, zorder=0)
        ax.axvline(0, color=PALETTE["grid"], linewidth=1.1, zorder=0)

        ax.plot(sub["delta_vul"], sub["delta_total"], color=PALETTE["bluegrey"], linewidth=1.8, zorder=1)

        for _, r in sub.iterrows():
            lam = float(r["lambda_fair"])
            ax.scatter(
                r["delta_vul"], r["delta_total"],
                s=110 if lam > 0 else 96,
                marker=marker_map[lam],
                color=color_map[lam],
                edgecolor="white", linewidth=1.2, zorder=3
            )
            lab = "效率基准" if lam == 0 else f"λ={lam:.1f}"
            dx, dy = label_offset_map[lam]
            ax.annotate(
                lab,
                xy=(r["delta_vul"], r["delta_total"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9.6,
                color=PALETTE["ink"],
            )

        max_abs_x = max(abs(sub["delta_vul"]).max(), 1.0)
        max_abs_y = max(abs(sub["delta_total"]).max(), 1.0)
        ax.set_xlim(-0.12 * max_abs_x, max_abs_x * 1.15)
        ax.set_ylim(min(sub["delta_total"].min() * 1.15, -0.12 * max_abs_y), max(sub["delta_total"].max() * 1.15, 0.12 * max_abs_y))
        ax.set_title(budget_label_inline(b), fontsize=12.5, pad=9)
        ax.grid(True, color=PALETTE["grid"], linewidth=0.85, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

    axes[0].set_ylabel("总效果变化（相对 λ=0）")
    axes[2].set_ylabel("总效果变化（相对 λ=0）")
    axes[2].set_xlabel("脆弱群体效果变化（相对 λ=0）")
    axes[3].set_xlabel("脆弱群体效果变化（相对 λ=0）")
    fig.suptitle(
        "图4  不同公平权重下的公平—效率权衡",
        x=0.01, y=0.99, ha="left",
        fontsize=16, fontweight="bold", color=PALETTE["ink"]
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    savefig(fig, os.path.join(outdir["figures"], "fig4_公平效率影响平面.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-dir",
        default=os.path.join(PROJECT_ROOT, "stage2_cost_mapping"),
        help="Stage 2 results directory",
    )
    parser.add_argument(
        "--stage3-dir",
        default=SCRIPT_DIR,
        help="Stage 3 results directory",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs"),
        help="Output directory",
    )
    parser.add_argument("--skip-optional-fairness", action="store_true", help="Skip Figure 4")
    args = parser.parse_args()

    set_global_style()
    outdir = ensure_dirs(args.output_dir)
    dfs = load_inputs(args.stage2_dir, args.stage3_dir)
    export_table1(dfs["mapping"], outdir)
    export_table2(dfs["menu"], dfs["summary"], outdir)
    plot_pareto_front(dfs["summary"], dfs["menu"], outdir)
    plot_uniform_vs_stratified_dumbbell(dfs["uvs"], outdir)
    plot_triptych_assignment_heatmap(dfs["assign0"], outdir)
    if not args.skip_optional_fairness:
        plot_fairness_efficiency_plane(dfs["eff"], dfs["fair"], outdir)


if __name__ == "__main__":
    main()
