import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTDIR = os.path.join(REPO_ROOT, "outputs", "optimization", "solver")
INPUT = os.path.join(REPO_ROOT, "outputs", "fse", "cost_mapping", "model3_stage2_cost_216_main.csv")

os.makedirs(OUTDIR, exist_ok=True)

Smap = {1: "低补贴", 2: "中补贴", 3: "高补贴"}
Lmap = {1: "短假期", 2: "中假期", 3: "长假期"}
Cmap = {1: "基础托育", 2: "增强托育"}

BIRTHS_2024_ZJ = 410000
BUDGETS = {"tight": 1.1e9, "moderate": 2.2e9, "ample": 4.4e9, "max": 8.9e9}
VULNERABLE = {"2-1-1", "3-1-1"}  # operationalized vulnerable set

def build_main():
    df = pd.read_csv(INPUT)
    df = df[df["scenario"] == "main_broad_q08_m012"].copy()
    df["N_g"] = df["pi_zj"] * BIRTHS_2024_ZJ
    df["is_vulnerable"] = df["group_id"].isin(VULNERABLE).astype(int)
    df["policy_label"] = df.apply(
        lambda r: f"{Smap[int(r['S'])]}–{Lmap[int(r['L'])]}–{Cmap[int(r['C'])]}",
        axis=1,
    )
    return df

def policy_summary(df):
    out = (
        df.groupby(["policy_id", "policy_label", "S", "L", "C"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "Cbar_zj": (x["C_total_zj"] * x["pi_zj"]).sum(),
                    "Ebar_zj": (x["delta_hat_zj"] * x["pi_zj"]).sum(),
                    "total_cost_yuan": (x["C_total_zj"] * x["pi_zj"] * BIRTHS_2024_ZJ).sum(),
                    "total_effect_units": (x["delta_hat_zj"] * x["pi_zj"] * BIRTHS_2024_ZJ).sum(),
                    "share_observed": x["n_obs_cell"].gt(0).mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    out["E_per_10k_yuan"] = np.where(
        out["Cbar_zj"] > 0,
        out["Ebar_zj"] / (out["Cbar_zj"] / 10000),
        np.nan,
    )
    return out

def add_pareto(summary):
    keep = []
    for i, row in summary.iterrows():
        dominated = False
        for j, row2 in summary.iterrows():
            if i == j:
                continue
            if (
                row2["Cbar_zj"] <= row["Cbar_zj"]
                and row2["Ebar_zj"] >= row["Ebar_zj"]
                and (
                    row2["Cbar_zj"] < row["Cbar_zj"]
                    or row2["Ebar_zj"] > row["Ebar_zj"]
                )
            ):
                dominated = True
                break
        keep.append(not dominated)
    summary = summary.copy()
    summary["is_pareto_front"] = keep
    return summary

def stratified_dp(df, budget, lam):
    groups = sorted(df["group_id"].unique())
    lookup = {(r["group_id"], r["policy_id"]): r for _, r in df.iterrows()}
    states = [(0.0, 0.0, {})]
    for g in groups:
        gsub = df[df["group_id"] == g]
        w = 1 + lam * (1 if g in VULNERABLE else 0)
        opts = [
            (
                r["policy_id"],
                float(r["N_g"] * r["C_total_zj"]),
                float(r["N_g"] * r["delta_hat_zj"] * w),
            )
            for _, r in gsub.iterrows()
        ]
        new = []
        for c0, u0, a0 in states:
            for pid, c, u in opts:
                nc = c0 + c
                if nc <= budget + 1e-6:
                    na = a0.copy()
                    na[g] = pid
                    new.append((nc, u0 + u, na))
        new.sort(key=lambda x: (round(x[0], 6), -x[1]))
        pruned = []
        best = -1e100
        for c, u, a in new:
            if u > best + 1e-9:
                pruned.append((c, u, a))
                best = u
        states = pruned

    best = max(states, key=lambda x: x[1])

    rows = []
    total_effect = 0.0
    vulnerable_effect = 0.0
    vulnerable_budget = 0.0
    for g, pid in best[2].items():
        r = lookup[(g, pid)]
        cost = float(r["N_g"] * r["C_total_zj"])
        eff = float(r["N_g"] * r["delta_hat_zj"])
        rows.append(
            {
                "group_id": g,
                "group_label": r["group_label"],
                "policy_id": pid,
                "policy_label": r["policy_label"],
                "S": int(r["S"]),
                "L": int(r["L"]),
                "C": int(r["C"]),
                "pi_zj": float(r["pi_zj"]),
                "N_g": float(r["N_g"]),
                "is_vulnerable": int(g in VULNERABLE),
                "delta_hat_zj": float(r["delta_hat_zj"]),
                "cost_per_household_yuan": float(r["C_total_zj"]),
                "group_total_cost_yuan": cost,
                "group_total_effect_units": eff,
            }
        )
        total_effect += eff
        if g in VULNERABLE:
            vulnerable_effect += eff
            vulnerable_budget += cost

    detail = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)
    counts = (
        detail["policy_id"]
        .value_counts()
        .rename_axis("policy_id")
        .reset_index(name="n_groups_assigned")
    )
    summary = {
        "total_cost_yuan": float(best[0]),
        "total_effect_units": float(total_effect),
        "weighted_utility": float(best[1]),
        "vulnerable_effect_units": float(vulnerable_effect),
        "vulnerable_budget_yuan": float(vulnerable_budget),
        "n_unique_policies": int(detail["policy_id"].nunique()),
        "frontier_states": int(len(states)),
    }
    return summary, detail, counts

def main():
    df = build_main()
    summary = add_pareto(policy_summary(df))
    summary.to_csv(os.path.join(OUTDIR, "model3_stage3_policy_summary_main.csv"), index=False)
    summary[summary["is_pareto_front"]].sort_values("Cbar_zj").to_csv(
        os.path.join(OUTDIR, "model3_stage3_pareto_front_main.csv"), index=False
    )

    # uniform candidates
    uniform_rows = []
    for lam in [0.0, 0.5, 1.0]:
        tmp = df.copy()
        tmp["omega"] = 1 + lam * tmp["is_vulnerable"]
        for pid, sub in tmp.groupby("policy_id"):
            uniform_rows.append(
                {
                    "lambda_fair": lam,
                    "policy_id": pid,
                    "policy_label": sub["policy_label"].iloc[0],
                    "total_cost_yuan": float((sub["N_g"] * sub["C_total_zj"]).sum()),
                    "total_effect_units": float((sub["N_g"] * sub["delta_hat_zj"]).sum()),
                    "weighted_utility": float((sub["N_g"] * sub["delta_hat_zj"] * sub["omega"]).sum()),
                }
            )
    pd.DataFrame(uniform_rows).to_csv(
        os.path.join(OUTDIR, "model3_stage3_uniform_policy_all_candidates.csv"), index=False
    )

    print("Stage 3 materialized to:", OUTDIR)

if __name__ == "__main__":
    main()
