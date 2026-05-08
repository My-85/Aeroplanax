# Planax 气动 Bug 终极诊断报告 (基于 NASA TP-1538 + 数据文件验证)

**日期**: 2026-05-08

---

## 🎯 最终结论

经过三层独立验证（NASA Fortran 代码 + Planax 数据文件实测 + 物理意义检查），现做出**确凿结论**：

### ✅ Planax 主气动表 (Bug A 不存在)

数据 reshape 顺序 `(DH, BETA, ALPHA)` **是正确的**，这与"文件名 ALPHA1_BETA1_DH1"暗示的相反。

**验证**：用 NASA TP-1538 原始 12×5 Cx 表（f16_deq.f line 294-303）作 ground truth 对比，
Planax 现状 reshape 给出的 Cx 值在 α∈[-10,15]°、EL∈[-25,25]° 范围内 RMSE = 0.005~0.075，**精度极佳**。
而错误的 (ALPHA, BETA, DH) reshape RMSE = 0.075~0.227，**明显错位**。

→ **`aero_data.py` 中所有 `reshape((DH, BETA, ALPHA))` 是对的，不要动**。

### ✅ 主系数调用 (Bug 不存在)

```python
Cx = hifi_F16._Cx((el, beta, alpha))    # ← 在 dynamics.py line 255 是对的
Cz = hifi_F16._Cz((el, beta, alpha))    # ← 对的
Cm = hifi_F16._Cm((el, beta, alpha))    # ← 对的
Cn = hifi_F16._Cn((el, beta, alpha))    # ← 对的
Cl = hifi_F16._Cl((el, beta, alpha))    # ← 对的
Cy = hifi_F16._Cy((beta, alpha))        # ← 对的
```

→ **dynamics.py 第 255-260 行是对的**。

### ❌ LEF / a20 / r30 修正项调用 (Bug 真实存在)

```python
# dynamics.py line 272-301 (BUGGY)
delta_Cx_lef = hifi_F16._Cx_lef((alpha, beta)) - hifi_F16._Cx((alpha, beta, 0))
                                 ^^^^^^^^^^^                  ^^^^^^^^^^^^^^^
                                 错：应是 (beta, alpha)        错：应是 (0, beta, alpha)
```

**证据**：
1. 函数 `_Cx_lef` 内部签名 `bilinear_interp(BETA1, ALPHA2, ...)` 期望 `(beta, alpha)`
2. 函数 `_Cx` 内部签名 `trilinear_interp(DH1, BETA1, ALPHA1, ...)` 期望 `(el, beta, alpha)`
3. NASA TP-1538 Fortran 函数签名都是 `(ALPHA, BETA, ...)` — 但 Planax **内部已经把数据按 (DH,BETA,ALPHA) 排好了**（与 NASA Fortran 的 `A(K_alpha, M_beta)` Fortran column-major 等价于 disk-row-major 的 `[BETA, ALPHA]`）。所以**调用接口的参数顺序和 NASA Fortran 表面相反**，但**底层物理是同一回事**。
4. **物理验证**：用错误调用计算 `delta_Cx_lef`，全 alpha 范围都为正且单调增（"LEF 部署在大 alpha 时反而大幅增阻"），违反 LEF 设计目的。用语义正确的调用，`delta_Cx_lef` 在 alpha~0 时近 0，大 alpha 时变负（"LEF 在大 alpha 时减阻"），符合物理。

→ **`dynamics.py` 第 272-301 行 LEF/a20/r30 部分确实有 bug，应修复**。

### ⚠️ 推力换算 (确实异常，但不一定是错)

```python
T = throttle * 0.225 * 76300 / 0.3048    # 全油门 ≈ 56,324 lbf
```

- 真实 F-100-PW-229 加力推力: 29,000 lbf
- NASA TP-1538 引擎模型最大: 20,000 lbf (海平面、零马赫)

Planax 给出的推力是真实的 1.94~2.8 倍。

但这**可能是有意为之**：Planax 是 RL 训练用的简化模型，可能为了让 policy 容易学到大机动特意放大推力。

→ **不要擅自修改推力**，除非你（dqy）明确说要修。

---

## 📊 总结表

| 怀疑点 | 实际是否有 bug | 修复建议 |
|------|------|------|
| 数据 reshape 顺序 | ❌ 不是 bug | 不动 |
| 主系数 `_Cx((el,beta,alpha))` 调用 | ❌ 不是 bug | 不动 |
| **LEF/a20/r30 调用** `((alpha,beta))` 和 `((alpha,beta,0))` | ✅ **真 bug** | **修复** |
| 推力换算 `* 0.225 * 76300 / 0.3048` | ⚠️ 数值异常但可能有意 | 不动（除非 dqy 说要修） |

---

## 🛠️ 真正应该做的修复

只修一处：**`dynamics.py` 第 272-301 行**的 LEF/a20/r30 调用参数顺序。

修改：
- `_X_lef((alpha, beta))` → `_X_lef((beta, alpha))`
- `_X((alpha, beta, 0))` → `_X((0.0, beta, alpha))`

不动：
- 数据 reshape (`aero_data.py` 全部不动)
- 主系数调用 (dynamics.py line 255-260)
- 推力换算 (dynamics.py line 379)

这样 RL 训练管线的**主行为基本不变**（因为推力和主气动都没动），只是 LEF 修正项变得物理正确。这个修改影响相对较小，更安全。
