# 建模三 Stage 0：数据处理与样本审计

## 1. 原始数据规模
- `clean_fse_long_main.csv`：6204 行，1034 名受访者。
- `clean_wide_main.csv`：1034 行，1034 名受访者。

## 2. 长表清洗规则
保留以下字段均非缺失的情景观测：
`rid, version, t, scene_code, P, I, G, S, L, C, rating, prov, city`。

## 3. 清洗结果
- 全样本 row-valid 长表：6149 行，1029 名受访者。
- 全样本 balanced 长表（每位受访者 6 个有效情景）：6072 行，1012 名受访者。
- 浙江 row-valid 长表：5057 行，845 名受访者。
- 浙江 balanced 长表：5022 行，837 名受访者。

## 4. 缺失说明
- 被完全剔除的受访者（6 个情景均无有效评分/编码）：['A_479', 'B_806', 'B_904', 'B_905', 'B_930']
- 仍有部分有效情景、但不足 6 个的受访者：['A_12', 'A_14', 'A_17', 'A_212', 'A_27', 'A_270', 'A_271', 'A_357', 'A_406', 'A_452', 'A_485', 'A_488', 'A_537', 'A_545', 'A_549', 'A_550', 'A_66']
- 在浙江子样本中，完整平衡样本与 row-valid 样本仅差 35 行，因此后续 FE 主估计可优先使用 balanced 浙江长表。

## 5. 浙江样本覆盖度
- 浙江 balanced 长表中共有 167 个被实际观测到的情景单元，完整设计网格为 216 个。
- 每个已观测情景单元的样本量：最小 4，中位数 34，最大 73。
- 每个政策包在浙江 balanced 样本中的观测次数：最小 229，最大 335。

## 6. 浙江处境权重与托育使用率口径
为与建模三中的 `g=(P,I,G)` 对接，采用 **浙江已婚女性样本** 构造实际处境权重与 `u_g`：
- `P_actual`：0 个子女→1；1 个子女→2；2 个及以上→3。
- `I_actual`：家庭月收入 `8000元及以下/8001-15000元` → 1；`15001元及以上` → 2。
- `G_actual`：`不能` → 1；`能` → 2。
- `u_g`：若 0–3 岁照料偏好为“长辈辅助+部分托育”或“托育机构照料”，记为 1；若为“长辈全职照料”，记为 0。

对应样本量：
- 浙江已婚女性样本：384。
- 可直接用于 `π_g^{ZJ}` 与 `u_g` 估计的完整处境样本：384。

## 7. 已输出文件
- `model3_long_rowvalid_all.csv`
- `model3_long_balanced_all.csv`
- `model3_long_rowvalid_zhejiang.csv`
- `model3_long_balanced_zhejiang.csv`
- `model3_wide_zhejiang_raw.csv`
- `model3_zhejiang_group_weights.csv`
- `model3_zhejiang_ug_by_group.csv`
- `model3_zhejiang_actual_group_sample_married_female.csv`
- `model3_zhejiang_observed_cell_counts.csv`
- `model3_zhejiang_full_216_design_grid.csv`
- `model3_zhejiang_policy_counts.csv`
- `model3_zhejiang_group_counts_long.csv`

## 8. 下一步
Stage 1 直接进入浙江子样本的固定效应估计与 $E_g^{ZJ}(p)=Δ_g^{ZJ}(p)$ 计算。
