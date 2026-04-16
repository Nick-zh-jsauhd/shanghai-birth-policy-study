# Stage 1：浙江子样本固定效应估计

主样本口径：balanced；稳健性口径：row-valid。

模型：受访者固定效应 + 家庭处境主效应 + 政策主效应 + 政策内部两两交互 + 家庭处境×政策两两交互。

输出文件包括：
- model3_stage1_cell_rankings_balanced_main.csv
- model3_stage1_cell_rankings_rowvalid_robust.csv
- model3_stage1_coefficients_full_balanced_main.csv
- model3_stage1_coefficients_full_rowvalid_robust.csv
- model3_stage1_coefficients_selected_balanced_main.csv
- model3_stage1_coefficients_selected_rowvalid_robust.csv
- model3_stage1_fit_stats_balanced_main.json
- model3_stage1_fit_stats_rowvalid_robust.json
- model3_stage1_mu_delta_216_balanced_main.csv
- model3_stage1_mu_delta_216_rowvalid_robust.csv
- model3_stage1_observed_fit_balanced_main.csv
- model3_stage1_observed_fit_rowvalid_robust.csv
- model3_stage1_policy_effects_18_balanced_main.csv
- model3_stage1_policy_effects_18_rowvalid_robust.csv
- model3_stage1_policy_effects_main_vs_robust.csv

## 主样本拟合统计
- nobs: 5022
- nparams: 911
- r2: 0.6094444543907511
- adj_r2: 0.5229921200428026
- rmse: 1.3481113870866155
- mae: 1.0186178455037158
- istop: 2
- iterations: 86
- normr: 95.53535709268427
- normar: 95.53535709268427
- conda: 7006.805551256702
- n_unique_rid: 837
- n_unique_scene_observed: 167
- n_full_grid: 216
- n_unobserved_grid_cells: 49
- pred_mu_min: 6.155491309140284
- pred_mu_max: 8.40081231336468
- pred_delta_min: -0.3101630536712632
- pred_delta_max: 1.3827130659770503

## 主样本政策效果前五
- 3-3-2: Ebar_zj=0.8621, mu_bar_zj=8.1533
- 3-3-1: Ebar_zj=0.8376, mu_bar_zj=8.1288
- 3-2-1: Ebar_zj=0.6391, mu_bar_zj=7.9303
- 2-3-2: Ebar_zj=0.5940, mu_bar_zj=7.8852
- 3-1-2: Ebar_zj=0.4883, mu_bar_zj=7.7796

## 主样本政策效果后五
- 2-1-2: Ebar_zj=0.2504, mu_bar_zj=7.5417
- 1-1-2: Ebar_zj=0.1019, mu_bar_zj=7.3931
- 1-2-2: Ebar_zj=0.0724, mu_bar_zj=7.3636
- 1-1-1: Ebar_zj=0.0000, mu_bar_zj=7.2912
- 2-1-1: Ebar_zj=-0.0348, mu_bar_zj=7.2564
