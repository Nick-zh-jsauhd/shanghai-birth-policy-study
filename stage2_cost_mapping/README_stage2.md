# Stage 2：浙江口径货币化成本映射

本目录给出建模三 Stage 2 的可复现结果。核心目标是把 Stage 1 的浙江样本政策效果
`E_g^{ZJ}(p)=Δ_g^{ZJ}(p)` 映射为浙江公共部门视角下的近似真实支出 `C_g^{ZJ}(p)`。

## 一、主分析口径

### 1. 现金补贴
- S1 = 0 元（相对浙江现行基线的增量）
- S2 = 10,800 元
- S3 = 21,600 元

### 2. 托育减负
- C1 = 0 元/年
- C2 = 8,400 元/年
- 组别成本：`C_C^{ZJ} = u_g × 8400`
- 其中 `u_g` 来自浙江样本 0–3 岁照料偏好估计

### 3. 假期支持（主分析）
主分析采用**增量化的广义公共支出口径**：

`C_L^{ZJ}(g,L) = q_g · [ r_g · \bar w_g · m_L + 0.5 · SC_g · k_L ]`

其中：
- `q_g = 0.8`
- `r_g = 1.0`
- `SC_g = τ_SI · \bar w_g`
- `τ_SI = 0.26`（模型校准参数）
- 月工资锚点：
  - 中低收入组：浙江 2024 私营单位平均工资 / 12 = 6435.83 元/月
  - 中高收入组：浙江 2024 非私营单位平均工资 / 12 = 11436.58 元/月
- 主分析桥接参数：
  - L1：`m_L=0, k_L=0`
  - L2：`m_L=1, k_L=6`
  - L3：`m_L=2, k_L=6`

> 注：这里对 `k_L` 采用了**增量化处理**，使低支持参照组 `L1` 的增量成本为 0，从而与
> Stage 1 中所有效果都相对参照组 `p0=(1,1,1)` 的设定保持一致。

## 二、稳健性场景
已同时输出：
- `broad_q06_m012`
- `broad_q10_m012`
- `broad_q08_m_conservative`
- `broad_q08_m_expanded`
- `narrow_fiscal_q08_m012`
- `main_broad_q08_m012_ug_shrunk`

## 三、核心输出文件
- `model3_stage2_policy_mapping_table.csv`：实验属性 → 浙江货币化映射表
- `model3_stage2_group_cost_inputs_main.csv`：各家庭处境组的成本输入参数
- `model3_stage2_cost_216_main.csv`：216 个 `(g,p)` 单元的主口径成本表
- `model3_stage2_policy_costs_18_main.csv`：18 个政策包的加权平均成本表（主口径）
- `model3_stage2_policy_costs_18_all_scenarios.csv`：18 个政策包的全部稳健性场景结果
- `model3_stage2_selected_policy_scenario_compare.csv`：代表性政策包的场景对比
- `model3_stage2_parameters_main.csv`：主口径参数总表

## 四、主口径下若干直接可用结果
- 基线预算锚点：`41.0万人 × 3600元 × 3 = 4,428,000,000 元/年`
- 18 政策包中，主口径下**总体效果最高**的是 `3-3-2`，
  加权平均成本约 `49793.20` 元/户，效果 `Ē=0.862`
- 在剔除基准组后，主口径下**单位万元效果最高**的是 `1-2-1`，
  每万元对应的评分增益约 `0.253`

## 五、下一步
Stage 3 可直接读取：
- `model3_stage2_policy_costs_18_main.csv`
- `model3_stage1_policy_effects_18_balanced_main.csv`
来构造浙江口径的成本—效果前沿、预算菜单以及统一投放/分层投放比较。
