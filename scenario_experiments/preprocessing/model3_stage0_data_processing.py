
import pandas as pd
import numpy as np
import os

outdir = "./model3_stage0"
os.makedirs(outdir, exist_ok=True)

long = pd.read_csv("clean_fse_long_main.csv")
wide = pd.read_csv("clean_wide_main.csv")

req = ['rid','version','t','scene_code','P','I','G','S','L','C','rating','prov','city']
long_rowvalid = long.dropna(subset=req).copy()
for c in ['t','scene_code','P','I','G','S','L','C','rating']:
    long_rowvalid[c] = long_rowvalid[c].astype(int)

map_P_label = {1:'首胎/无子女',2:'二胎/已有1孩',3:'三胎/已有2孩及以上'}
map_I_label = {1:'中低收入',2:'中高收入'}
map_G_label = {1:'无祖辈照护',2:'有祖辈照护'}
map_S_label = {1:'低补贴',2:'中补贴',3:'高补贴'}
map_L_label = {1:'短假期',2:'中假期',3:'长假期'}
map_C_label = {1:'基础托育',2:'增强托育'}

for c, mp, newc in [
    ('P', map_P_label, 'P_label'),
    ('I', map_I_label, 'I_label'),
    ('G', map_G_label, 'G_label'),
    ('S', map_S_label, 'S_label'),
    ('L', map_L_label, 'L_label'),
    ('C', map_C_label, 'C_label'),
]:
    long_rowvalid[newc] = long_rowvalid[c].map(mp)

long_rowvalid['group_id'] = long_rowvalid[['P','I','G']].astype(str).agg('-'.join, axis=1)
long_rowvalid['policy_id'] = long_rowvalid[['S','L','C']].astype(str).agg('-'.join, axis=1)
long_rowvalid['group_label'] = long_rowvalid['P_label']+'|'+long_rowvalid['I_label']+'|'+long_rowvalid['G_label']
long_rowvalid['policy_label'] = long_rowvalid['S_label']+'-'+long_rowvalid['L_label']+'-'+long_rowvalid['C_label']

full6 = long_rowvalid.groupby('rid').size()
full6_rids = full6[full6==6].index
long_balanced = long_rowvalid[long_rowvalid['rid'].isin(full6_rids)].copy()

zj_rowvalid = long_rowvalid[long_rowvalid['prov']=='浙江'].copy()
zj_balanced = long_balanced[long_balanced['prov']=='浙江'].copy()

long_rowvalid.to_csv(os.path.join(outdir,'model3_long_rowvalid_all.csv'), index=False, encoding='utf-8-sig')
long_balanced.to_csv(os.path.join(outdir,'model3_long_balanced_all.csv'), index=False, encoding='utf-8-sig')
zj_rowvalid.to_csv(os.path.join(outdir,'model3_long_rowvalid_zhejiang.csv'), index=False, encoding='utf-8-sig')
zj_balanced.to_csv(os.path.join(outdir,'model3_long_balanced_zhejiang.csv'), index=False, encoding='utf-8-sig')

wide_zj = wide[wide['prov']=='浙江'].copy()
wide_zj.to_csv(os.path.join(outdir,'model3_wide_zhejiang_raw.csv'), index=False, encoding='utf-8-sig')

sex_col='您的性别:'
marital_col='您目前的婚育状态？'
child_col='您目前的子女数量:'
hhinc_col='家庭月总收入水平:'
help_col='您和伴侣双方父母是否至少一方能够提供日常育儿帮助?'
care_col='4.\t您更倾向哪种0-3岁婴幼儿照料方式?'

female = wide_zj[wide_zj[sex_col]=='女'].copy()
married_female = female[female[marital_col]=='已婚'].copy()

def map_P(x):
    if x == '0 个':
        return 1
    elif x == '1 个':
        return 2
    elif x in ['2 个','3 个及以上']:
        return 3
    return np.nan

def map_I(x):
    if x in ['8000元及以下','8001-15000元']:
        return 1
    elif x in ['15001-25000元','25001-40000元','40001元以上']:
        return 2
    return np.nan

def map_G(x):
    if x == '不能':
        return 1
    elif x == '能':
        return 2
    return np.nan

married_female['P_actual'] = married_female[child_col].map(map_P)
married_female['I_actual'] = married_female[hhinc_col].map(map_I)
married_female['G_actual'] = married_female[help_col].map(map_G)
married_female['group_id'] = married_female[['P_actual','I_actual','G_actual']].astype('Int64').astype(str).agg('-'.join, axis=1)
married_female['formal_childcare_pref'] = married_female[care_col].isin([' 长辈辅助+部分托育',' 托育机构照料']).astype(int)

actual_group = married_female.dropna(subset=['P_actual','I_actual','G_actual']).copy()
actual_group['P_actual'] = actual_group['P_actual'].astype(int)
actual_group['I_actual'] = actual_group['I_actual'].astype(int)
actual_group['G_actual'] = actual_group['G_actual'].astype(int)
actual_group['P_label'] = actual_group['P_actual'].map(map_P_label)
actual_group['I_label'] = actual_group['I_actual'].map(map_I_label)
actual_group['G_label'] = actual_group['G_actual'].map(map_G_label)
actual_group['group_label'] = actual_group['P_label']+'|'+actual_group['I_label']+'|'+actual_group['G_label']

group_weights = (
    actual_group.groupby(['P_actual','I_actual','G_actual','group_id','group_label'])
    .size().rename('n').reset_index()
)
group_weights['pi_zj'] = group_weights['n'] / group_weights['n'].sum()

ug = (
    actual_group.groupby(['P_actual','I_actual','G_actual','group_id','group_label'])['formal_childcare_pref']
    .agg(ug='mean', n_group='size', n_formal='sum')
    .reset_index()
)

group_weights.to_csv(os.path.join(outdir,'model3_zhejiang_group_weights.csv'), index=False, encoding='utf-8-sig')
ug.to_csv(os.path.join(outdir,'model3_zhejiang_ug_by_group.csv'), index=False, encoding='utf-8-sig')
actual_group.to_csv(os.path.join(outdir,'model3_zhejiang_actual_group_sample_married_female.csv'), index=False, encoding='utf-8-sig')

obs_cell = (
    zj_balanced.groupby(['P','I','G','S','L','C','group_id','group_label','policy_id','policy_label'])
    .size().rename('n_obs').reset_index()
)
obs_cell.to_csv(os.path.join(outdir,'model3_zhejiang_observed_cell_counts.csv'), index=False, encoding='utf-8-sig')

grid = pd.MultiIndex.from_product(
    [[1,2,3],[1,2],[1,2],[1,2,3],[1,2,3],[1,2]],
    names=['P','I','G','S','L','C']
).to_frame(index=False)

for c, mp, newc in [
    ('P', map_P_label, 'P_label'),
    ('I', map_I_label, 'I_label'),
    ('G', map_G_label, 'G_label'),
    ('S', map_S_label, 'S_label'),
    ('L', map_L_label, 'L_label'),
    ('C', map_C_label, 'C_label'),
]:
    grid[newc] = grid[c].map(mp)

grid['group_id'] = grid[['P','I','G']].astype(str).agg('-'.join, axis=1)
grid['policy_id'] = grid[['S','L','C']].astype(str).agg('-'.join, axis=1)
grid['group_label'] = grid['P_label']+'|'+grid['I_label']+'|'+grid['G_label']
grid['policy_label'] = grid['S_label']+'-'+grid['L_label']+'-'+grid['C_label']
grid = grid.merge(obs_cell[['P','I','G','S','L','C','n_obs']], on=['P','I','G','S','L','C'], how='left')
grid['n_obs'] = grid['n_obs'].fillna(0).astype(int)
grid['is_observed'] = (grid['n_obs']>0).astype(int)
grid.to_csv(os.path.join(outdir,'model3_zhejiang_full_216_design_grid.csv'), index=False, encoding='utf-8-sig')
