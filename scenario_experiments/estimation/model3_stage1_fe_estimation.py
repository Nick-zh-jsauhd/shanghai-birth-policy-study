import os
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

BASE = '/mnt/data/model3_stage0'
OUT = '/mnt/data/model3_stage1'
os.makedirs(OUT, exist_ok=True)

MAIN_FILE = os.path.join(BASE, 'model3_long_balanced_zhejiang.csv')
ROB_FILE = os.path.join(BASE, 'model3_long_rowvalid_zhejiang.csv')
WEIGHT_FILE = os.path.join(BASE, 'model3_zhejiang_group_weights.csv')

MAIN = pd.read_csv(MAIN_FILE)
ROB = pd.read_csv(ROB_FILE)
WEIGHTS = pd.read_csv(WEIGHT_FILE)

# ---------- helpers ----------
BASELINE_POLICY = {'S':1, 'L':1, 'C':1}

def prepare_df(df):
    df = df.copy()
    # Keep canonical variable names as strings for categorical processing
    for c in ['rid','P','I','G','S','L','C']:
        df[c] = df[c].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    return df

TERM_SPECS = [
    ('P', ['P']),
    ('I', ['I']),
    ('G', ['G']),
    ('S', ['S']),
    ('L', ['L']),
    ('C', ['C']),
    ('SL', ['S','L']),
    ('SC', ['S','C']),
    ('LC', ['L','C']),
    ('PS', ['P','S']),
    ('PL', ['P','L']),
    ('PC', ['P','C']),
    ('IS', ['I','S']),
    ('IL', ['I','L']),
    ('IC', ['I','C']),
    ('GS', ['G','S']),
    ('GL', ['G','L']),
    ('GC', ['G','C']),
    ('rid', ['rid'])
]

def make_term_series(df, cols):
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + ':' + df[c].astype(str)
    return out

class SparseCategoricalOLS:
    def __init__(self, term_specs):
        self.term_specs = term_specs
        self.term_levels = {}
        self.term_colnames = []
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None

    def fit(self, df, ycol='rating'):
        y = df[ycol].to_numpy(dtype=float)
        X_blocks = [sparse.csr_matrix(np.ones((len(df),1), dtype=float))]
        feature_names = ['Intercept']
        self.term_levels = {}

        for tname, cols in self.term_specs:
            s = make_term_series(df, cols)
            levels = sorted(pd.unique(s))
            self.term_levels[tname] = levels
            level_to_idx = {lev:i for i, lev in enumerate(levels[1:])}  # drop first level
            rows, cols_idx, data = [], [], []
            for r, lev in enumerate(s):
                idx = level_to_idx.get(lev)
                if idx is not None:
                    rows.append(r); cols_idx.append(idx); data.append(1.0)
            ncols = max(len(levels)-1, 0)
            if ncols > 0:
                mat = sparse.csr_matrix((data, (rows, cols_idx)), shape=(len(df), ncols), dtype=float)
                X_blocks.append(mat)
                feature_names.extend([f'{tname}[{lev}]' for lev in levels[1:]])

        X = sparse.hstack(X_blocks, format='csr')
        # Solve OLS via sparse least squares
        sol = lsqr(X, y, atol=1e-10, btol=1e-10)
        beta = sol[0]
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        self.feature_names_ = feature_names
        self._X_fit_shape = X.shape
        self._fit_stats = {
            'istop': int(sol[1]),
            'iterations': int(sol[2]),
            'normr': float(sol[3]),
            'normar': float(sol[4]),
            'conda': float(sol[6])
        }
        yhat = X @ beta
        resid = y - yhat
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        r2 = 1 - ss_res/ss_tot
        n, k = X.shape
        adj_r2 = 1 - (1-r2)*(n-1)/(n-k) if n > k else np.nan
        rmse = float(np.sqrt(np.mean(resid**2)))
        mae = float(np.mean(np.abs(resid)))
        self._perf = {'nobs':int(n),'nparams':int(k),'r2':float(r2),'adj_r2':float(adj_r2),'rmse':rmse,'mae':mae}
        return self

    def design_matrix(self, df):
        X_blocks = [sparse.csr_matrix(np.ones((len(df),1), dtype=float))]
        for tname, cols in self.term_specs:
            s = make_term_series(df, cols)
            levels = self.term_levels[tname]
            level_to_idx = {lev:i for i, lev in enumerate(levels[1:])}
            rows, cols_idx, data = [], [], []
            for r, lev in enumerate(s):
                idx = level_to_idx.get(lev)
                if idx is not None:
                    rows.append(r); cols_idx.append(idx); data.append(1.0)
            ncols = max(len(levels)-1, 0)
            if ncols > 0:
                mat = sparse.csr_matrix((data, (rows, cols_idx)), shape=(len(df), ncols), dtype=float)
                X_blocks.append(mat)
        return sparse.hstack(X_blocks, format='csr')

    def predict(self, df):
        X = self.design_matrix(df)
        beta = np.concatenate([[self.intercept_], self.coef_])
        return np.asarray(X @ beta).reshape(-1)


def build_full_grid(rids):
    rows = []
    for P in [1,2,3]:
        for I in [1,2]:
            for G in [1,2]:
                for S in [1,2,3]:
                    for L in [1,2,3]:
                        for C in [1,2]:
                            for rid in rids:
                                rows.append((str(rid), str(P), str(I), str(G), str(S), str(L), str(C)))
    return pd.DataFrame(rows, columns=['rid','P','I','G','S','L','C'])


def stage1_run(df_raw, tag):
    df = prepare_df(df_raw)
    # fit
    model = SparseCategoricalOLS(TERM_SPECS).fit(df, ycol='rating')

    # predictions on observed data
    df_out = df[['rid','scene_code','P','I','G','S','L','C','rating']].copy()
    df_out['mu_hat_obs'] = model.predict(df)
    df_out['resid'] = df_out['rating'] - df_out['mu_hat_obs']
    df_out.to_csv(os.path.join(OUT, f'model3_stage1_observed_fit_{tag}.csv'), index=False)

    # selected coefficients: exclude rid FE for readability
    coef_df = pd.DataFrame({'term': model.feature_names_, 'coef': np.concatenate([[model.intercept_], model.coef_])})
    coef_df_norid = coef_df[~coef_df['term'].str.startswith('rid[')].copy()
    coef_df_norid.to_csv(os.path.join(OUT, f'model3_stage1_coefficients_selected_{tag}.csv'), index=False)
    coef_df.to_csv(os.path.join(OUT, f'model3_stage1_coefficients_full_{tag}.csv'), index=False)

    # full 216-grid averaged across rid FE distribution
    rids = sorted(df['rid'].unique())
    full = build_full_grid(rids)
    full['mu_hat'] = model.predict(full)
    grid = (full.groupby(['P','I','G','S','L','C'], as_index=False)
                .agg(mu_hat_zj=('mu_hat','mean')))
    # baseline within each (P,I,G)
    base = grid[(grid['S']=='1') & (grid['L']=='1') & (grid['C']=='1')][['P','I','G','mu_hat_zj']].rename(columns={'mu_hat_zj':'mu_hat_base_zj'})
    grid = grid.merge(base, on=['P','I','G'], how='left')
    grid['delta_hat_zj'] = grid['mu_hat_zj'] - grid['mu_hat_base_zj']
    for c in ['P','I','G','S','L','C']:
        grid[c] = grid[c].astype(int)
    grid['group_id'] = grid['P'].astype(str)+'-'+grid['I'].astype(str)+'-'+grid['G'].astype(str)
    grid['policy_id'] = grid['S'].astype(str)+'-'+grid['L'].astype(str)+'-'+grid['C'].astype(str)
    obs_counts = (df.groupby(['P','I','G','S','L','C'], as_index=False)
                    .size().rename(columns={'size':'n_obs_cell'}))
    for c in ['P','I','G','S','L','C']:
        obs_counts[c] = obs_counts[c].astype(int)
    grid = grid.merge(obs_counts, on=['P','I','G','S','L','C'], how='left')
    grid['n_obs_cell'] = grid['n_obs_cell'].fillna(0).astype(int)
    grid.to_csv(os.path.join(OUT, f'model3_stage1_mu_delta_216_{tag}.csv'), index=False)

    # aggregate by policy using 浙江处境权重
    w = WEIGHTS[['group_id','pi_zj']].copy()
    pol = grid.merge(w, on='group_id', how='left')
    pol['weighted_delta'] = pol['delta_hat_zj'] * pol['pi_zj']
    pol_agg = (pol.groupby(['S','L','C','policy_id'], as_index=False)
                 .agg(Ebar_zj=('weighted_delta','sum'),
                      mu_bar_zj=('mu_hat_zj', lambda s: float(np.average(s, weights=pol.loc[s.index, 'pi_zj']))),
                      share_observed=('n_obs_cell', lambda s: float(np.mean(s>0)))))
    pol_agg = pol_agg.sort_values(['Ebar_zj','mu_bar_zj'], ascending=[False,False]).reset_index(drop=True)
    pol_agg.to_csv(os.path.join(OUT, f'model3_stage1_policy_effects_18_{tag}.csv'), index=False)

    # top/bottom and diagnostics
    perf = model._perf.copy()
    perf.update(model._fit_stats)
    perf['n_unique_rid'] = int(df['rid'].nunique())
    perf['n_unique_scene_observed'] = int(df[['P','I','G','S','L','C']].drop_duplicates().shape[0])
    perf['n_full_grid'] = 216
    perf['n_unobserved_grid_cells'] = int((grid['n_obs_cell']==0).sum())
    perf['pred_mu_min'] = float(grid['mu_hat_zj'].min())
    perf['pred_mu_max'] = float(grid['mu_hat_zj'].max())
    perf['pred_delta_min'] = float(grid['delta_hat_zj'].min())
    perf['pred_delta_max'] = float(grid['delta_hat_zj'].max())
    with open(os.path.join(OUT, f'model3_stage1_fit_stats_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(perf, f, ensure_ascii=False, indent=2)

    # readable summaries
    cell_rank = grid.sort_values('delta_hat_zj', ascending=False).reset_index(drop=True)
    cell_rank.to_csv(os.path.join(OUT, f'model3_stage1_cell_rankings_{tag}.csv'), index=False)
    return perf, grid, pol_agg, coef_df_norid

main_perf, main_grid, main_pol, main_coef = stage1_run(MAIN, 'balanced_main')
rob_perf, rob_grid, rob_pol, rob_coef = stage1_run(ROB, 'rowvalid_robust')

# comparison main vs robust at policy level
cmp = main_pol.merge(rob_pol, on='policy_id', suffixes=('_main','_robust'))
cmp['Ebar_diff_main_minus_robust'] = cmp['Ebar_zj_main'] - cmp['Ebar_zj_robust']
cmp['mu_diff_main_minus_robust'] = cmp['mu_bar_zj_main'] - cmp['mu_bar_zj_robust']
cmp.to_csv(os.path.join(OUT, 'model3_stage1_policy_effects_main_vs_robust.csv'), index=False)

# markdown summary
with open(os.path.join(OUT, 'README_stage1.md'), 'w', encoding='utf-8') as f:
    f.write('# Stage 1：浙江子样本固定效应估计\n\n')
    f.write('主样本口径：balanced；稳健性口径：row-valid。\n\n')
    f.write('模型：受访者固定效应 + 家庭处境主效应 + 政策主效应 + 政策内部两两交互 + 家庭处境×政策两两交互。\n\n')
    f.write('输出文件包括：\n')
    for fn in sorted(os.listdir(OUT)):
        if fn != 'README_stage1.md':
            f.write(f'- {fn}\n')
    f.write('\n## 主样本拟合统计\n')
    for k,v in main_perf.items():
        f.write(f'- {k}: {v}\n')
    f.write('\n## 主样本政策效果前五\n')
    for _, r in main_pol.head(5).iterrows():
        f.write(f"- {r['policy_id']}: Ebar_zj={r['Ebar_zj']:.4f}, mu_bar_zj={r['mu_bar_zj']:.4f}\n")
    f.write('\n## 主样本政策效果后五\n')
    for _, r in main_pol.tail(5).iterrows():
        f.write(f"- {r['policy_id']}: Ebar_zj={r['Ebar_zj']:.4f}, mu_bar_zj={r['mu_bar_zj']:.4f}\n")

print('DONE')
print('main_perf', json.dumps(main_perf, ensure_ascii=False))
print('top5_main')
print(main_pol.head(5).to_string(index=False))
print('bottom5_main')
print(main_pol.tail(5).to_string(index=False))
