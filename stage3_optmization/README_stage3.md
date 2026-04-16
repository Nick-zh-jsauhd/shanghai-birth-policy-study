# Stage 3：浙江口径成本—效果前沿、预算菜单与统一投放/分层投放优化

## 输入文件
- Stage 1: `model3_stage1_mu_delta_216_balanced_main.csv`
- Stage 2: `model3_stage2_cost_216_main.csv`

## 本阶段做了什么
1. 将 Stage 1 的浙江口径效果 `Δ_g^{ZJ}(p)` 与 Stage 2 的浙江口径成本 `C_g^{ZJ}(p)` 合并为完整的 216 单元表。
2. 在政策包总体层面计算：
   - 平均成本 `C̄^{ZJ}(p)=∑_g π_g C_g^{ZJ}(p)`
   - 平均效果 `Ē^{ZJ}(p)=∑_g π_g Δ_g^{ZJ}(p)`
3. 在 `最小化成本、最大化效果` 的准则下筛选浙江口径帕累托前沿。
4. 将总预算锁定为四档：
   - tight = 11 亿元
   - moderate = 22 亿元
   - ample = 44 亿元
   - max = 89 亿元
5. 令目标群体规模采用 `N_g = π_g × 41.0 万` 的年度出生 cohort 近似。
6. 求解两类优化问题：
   - 统一投放：所有处境组使用同一个政策包
   - 分层投放：允许不同处境组使用不同政策包
7. 报告两类目标：
   - 效率最优：`ω_g = 1`
   - 公平加权：`ω_g = 1 + λ·1(g∈V)`，其中 `V = {2-1-1, 3-1-1}`，即“较高孩次 + 中低收入 + 无祖辈照护”的脆弱组；`λ∈{0.5,1.0}`

## 关键输出
- `model3_stage3_pareto_front_main.csv`：浙江口径政策前沿
- `model3_stage3_frontier_representative_menu.csv`：代表性前沿菜单
- `model3_stage3_budget_compare_uniform_vs_stratified.csv`：预算下统一投放 vs 分层投放对比
- `model3_stage3_budget_compare_efficiency_main.csv`：效率最优主表
- `model3_stage3_budget_compare_fairness.csv`：公平加权对比表
- `model3_stage3_assignment_lambda*_*.csv`：各预算档位下的组别分配明细
- `model3_stage3_assignment_matrix_lambda0p0.csv`：效率最优分层投放矩阵
- `model3_stage3_assignment_matrix_lambda1p0.csv`：公平权重 λ=1 的分层投放矩阵

## 结果简记（效率最优，λ=0）
- 帕累托前沿共 8 个点（含基准组）：
  `1-1-1, 1-1-2, 1-2-1, 1-3-1, 3-1-2, 3-2-1, 3-3-1, 3-3-2`
- 统一投放在 11/22 亿元预算下只能选择基准组 `1-1-1`。
- 分层投放在 11/22 亿元预算下已经可以实现显著正向总效果。
- 44 亿元预算下，统一投放最优是 `1-1-2`；分层投放总效果比统一投放高约 229.29%。
- 89 亿元预算下，统一投放最优是 `1-2-1`；分层投放总效果比统一投放高约 70.50%。

## 口径提醒
- 这里的预算比较严格遵循 Stage 2 的“浙江年度出生 cohort × 增量支出”口径，因此统一投放会显得更难。
- 分层投放之所以优势明显，本质上来自“把高成本强支持包集中投向高响应处境组，把零成本或低成本包留给低响应组”。
