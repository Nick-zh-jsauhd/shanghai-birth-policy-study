import pandas as pd
import numpy as np
import json
import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_STAGE0_DIR = os.path.join(REPO_ROOT, 'outputs', 'fse', 'preprocessing')
DEFAULT_STAGE1_DIR = os.path.join(REPO_ROOT, 'outputs', 'fse', 'estimation')
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, 'outputs', 'fse', 'cost_mapping')

def build_stage2(stage0_dir=DEFAULT_STAGE0_DIR,
                 stage1_dir=DEFAULT_STAGE1_DIR,
                 outdir=DEFAULT_OUTDIR):
    os.makedirs(outdir, exist_ok=True)

    weights = pd.read_csv(f'{stage0_dir}/model3_zhejiang_group_weights.csv')
    ug_raw = pd.read_csv(f'{stage0_dir}/model3_zhejiang_ug_by_group.csv')
    mu216 = pd.read_csv(f'{stage1_dir}/model3_stage1_mu_delta_216_balanced_main.csv')

    group = (
        weights[['group_id','group_label','P_actual','I_actual','G_actual','pi_zj']]
        .merge(ug_raw[['group_id','ug','n_group','n_formal']], on='group_id', how='left')
    )
    overall_ug = ug_raw['n_formal'].sum()/ug_raw['n_group'].sum()
    prior_n = 10
    group['ug_shrunk'] = (group['n_formal'] + prior_n*overall_ug) / (group['n_group'] + prior_n)

    annual_private = 77230.0
    annual_nonprivate = 137239.0
    monthly_private = annual_private / 12.0
    monthly_nonprivate = annual_nonprivate / 12.0

    group['wage_anchor_annual'] = np.where(group['I_actual']==1, annual_private, annual_nonprivate)
    group['wage_anchor_monthly'] = np.where(group['I_actual']==1, monthly_private, monthly_nonprivate)
    group['income_anchor'] = np.where(group['I_actual']==1, '中低收入→私营单位平均工资', '中高收入→非私营单位平均工资')
    group.to_csv(f'{outdir}/model3_stage2_group_cost_inputs_main.csv', index=False)

    scenarios = [
        dict(scenario='main_broad_q08_m012', q=0.8, rg=1.0, tau_si=0.26, m_map={1:0,2:1,3:2}, k_map={1:0,2:6,3:6}, ug_col='ug', note='主结果：广义公共支出口径；q=0.8；m_L={0,1,2}'),
        dict(scenario='broad_q06_m012', q=0.6, rg=1.0, tau_si=0.26, m_map={1:0,2:1,3:2}, k_map={1:0,2:6,3:6}, ug_col='ug', note='假期实际使用率下界'),
        dict(scenario='broad_q10_m012', q=1.0, rg=1.0, tau_si=0.26, m_map={1:0,2:1,3:2}, k_map={1:0,2:6,3:6}, ug_col='ug', note='假期实际使用率上界'),
        dict(scenario='broad_q08_m_conservative', q=0.8, rg=1.0, tau_si=0.26, m_map={1:0,2:5/21.75,3:10/21.75}, k_map={1:0,2:6,3:6}, ug_col='ug', note='新增公共支付月数保守口径'),
        dict(scenario='broad_q08_m_expanded', q=0.8, rg=1.0, tau_si=0.26, m_map={1:0,2:2,3:4}, k_map={1:0,2:6,3:6}, ug_col='ug', note='新增公共支付月数扩展口径'),
        dict(scenario='narrow_fiscal_q08_m012', q=0.8, rg=0.0, tau_si=0.26, m_map={1:0,2:1,3:2}, k_map={1:0,2:6,3:6}, ug_col='ug', note='狭义财政口径：仅计企业社保补贴'),
        dict(scenario='main_broad_q08_m012_ug_shrunk', q=0.8, rg=1.0, tau_si=0.26, m_map={1:0,2:1,3:2}, k_map={1:0,2:6,3:6}, ug_col='ug_shrunk', note='托育使用率小样本收缩稳健性'),
    ]

    base = mu216.merge(group[['group_id','group_label','pi_zj','ug','ug_shrunk','wage_anchor_monthly','income_anchor']], on='group_id', how='left')
    all_cells, all_policies = [], []
    for sc in scenarios:
        df = base.copy()
        df['q_g'] = sc['q']
        df['r_g'] = sc['rg']
        df['tau_si'] = sc['tau_si']
        df['w_g'] = df['wage_anchor_monthly']
        df['m_L'] = df['L'].map(sc['m_map'])
        df['k_L'] = df['L'].map(sc['k_map'])
        df['C_cash_zj'] = df['S'].map({1:0.0, 2:10800.0, 3:21600.0})
        df['A_C'] = np.where(df['C']==2, 8400.0, 0.0)
        df['u_g_used'] = df[sc['ug_col']]
        df['C_childcare_zj'] = df['u_g_used'] * df['A_C']
        df['SC_g'] = df['tau_si'] * df['w_g']
        df['C_leave_birth_allowance'] = df['q_g'] * (df['r_g'] * df['w_g'] * df['m_L'])
        df['C_leave_socsub'] = df['q_g'] * (0.5 * df['SC_g'] * df['k_L'])
        df['C_leave_zj'] = df['C_leave_birth_allowance'] + df['C_leave_socsub']
        df['C_total_zj'] = df['C_cash_zj'] + df['C_childcare_zj'] + df['C_leave_zj']
        df['scenario'] = sc['scenario']
        df['scenario_note'] = sc['note']
        all_cells.append(df)

        pol = df.groupby(['scenario','scenario_note','policy_id','S','L','C'], as_index=False).apply(
            lambda x: pd.Series({
                'Cbar_zj': float((x['C_total_zj'] * x['pi_zj']).sum()),
                'Cbar_cash': float((x['C_cash_zj'] * x['pi_zj']).sum()),
                'Cbar_childcare': float((x['C_childcare_zj'] * x['pi_zj']).sum()),
                'Cbar_leave': float((x['C_leave_zj'] * x['pi_zj']).sum()),
                'Ebar_zj': float((x['delta_hat_zj'] * x['pi_zj']).sum()),
                'share_observed': float((x['n_obs_cell'] > 0).mean())
            })
        ).reset_index(drop=True)
        pol['E_per_10k_yuan'] = np.where(pol['Cbar_zj']>0, pol['Ebar_zj']/(pol['Cbar_zj']/10000.0), np.nan)
        all_policies.append(pol)

    cells = pd.concat(all_cells, ignore_index=True)
    policies = pd.concat(all_policies, ignore_index=True)
    cells.to_csv(f'{outdir}/model3_stage2_cost_216_all_scenarios.csv', index=False)
    policies.to_csv(f'{outdir}/model3_stage2_policy_costs_18_all_scenarios.csv', index=False)

    main_cells = cells[cells['scenario']=='main_broad_q08_m012'].copy()
    main_policies = policies[policies['scenario']=='main_broad_q08_m012'].copy().sort_values('Cbar_zj').reset_index(drop=True)
    main_cells.to_csv(f'{outdir}/model3_stage2_cost_216_main.csv', index=False)
    main_policies.to_csv(f'{outdir}/model3_stage2_policy_costs_18_main.csv', index=False)

if __name__ == '__main__':
    build_stage2()
