# NTTDA 的 AO-ETF 校正

## 推导与代码对应 — 2026-10-11

**目标：** 在独立 NAC 和 `compute_frame` 中使用同一个电子平移校正规则。
**假设：** 实轨道、实振幅、非简并根和收敛的 SCF/NTTDA；采用自旋适配
辅助波函数（AWF）及原子中心 AO 的一阶 built-in ETF 近似。
DFT 沿用固定积分网格、不含网格响应的导数模型。

令 $Q$ 为一个核笛卡尔坐标，$I,J$ 为两个根，$p,q$ 为 MO，$\mu,\nu$ 为 AO。
$X_I$ 为 NTTDA 振幅，$A$ 为激发矩阵，$C$ 为 MO 系数，$S$ 为 AO 重叠矩阵，
$\Delta\omega=\omega_J-\omega_I$ 为能隙。
子空间记号 C/O/V 分别表示闭壳层、开壳层和虚轨道；$\phi_p$ 表示 MO。
态间密度为 $\Gamma^{IJ}_{pq}=\langle\Psi_I|\sum_\sigma
a^\dagger_{p\sigma}a_{q\sigma}|\Psi_J\rangle$，自旋求和是显式的；
$D^{IJ}=C\Gamma^{IJ}C^T$。矩阵内积 $B:F=\sum_{pq}B_{pq}F_{pq}$ 不隐含转置。

### 1. 分开 AO 联络与 MO 响应

$$
T^Q_{\mu\nu}=\langle\chi_\mu|\partial_Q\chi_\nu\rangle,\qquad
S^{[Q]}=T^Q+T^{Q,T},\qquad S_A^Q=\frac{T^Q-T^{Q,T}}2 .
$$

→ **解释：** 完整重叠导数 $S^{[Q]}$ 是对称的，$S_A^Q$ 是反对称 AO 联络。
PySCF `get_ovlp` 给出对应原子的 bra 导数，`_ao_csf_connection` 中的转置和符号
将它转换为这里的 $D:S_A$ 收缩。

写 $\partial_Q C=C\kappa^Q$，定义 $\kappa_A^Q=(\kappa^Q-\kappa^{Q,T})/2$。
对 $C^TSC=1$ 微分可得

$$
\kappa^Q+\kappa^{Q,T}=-C^TS^{[Q]}C,\qquad
\langle\phi_p|\partial_Q\phi_q\rangle
=\left[\kappa_A^Q+C^TS_A^QC\right]_{pq} .
$$

→ **解释：** MO 正交条件消去 AO 联络的对称部分，剩下独立轨道旋转和反对称 AO 项。
对 C/O/V 内部旋转采用平行输运规范；不同子空间之间的响应由现有伴随方程求解。

$$
h_{IJ}^Q=X_I^T(\partial_Q A)X_J,\qquad
d_{IJ}^Q=\frac{h_{IJ}^Q}{\Delta\omega}
+\underbrace{\Gamma^{IJ}:\kappa_A^Q}_{d_{IJ,\mathrm{orb}}^Q}
+\underbrace{D^{IJ}:S_A^Q}_{d_{IJ,\mathrm{AO}}^Q} .
$$

→ **解释：** 微分本征方程得到振幅导数 $X_I^T\partial_QX_J=h_{IJ}^Q/\Delta\omega$。
组态基本身的导数另给出后两项。`hf_numerator` 已包含激发矩阵的轨道松弛，
仍需添加移动组态基的响应；两者不是重复计数。

### 2. ETF 只消去 AO 项

给原子 $A$ 上的 AO 乘上电子平移相位
$\exp(i m_e\mathbf v_A\cdot\mathbf r/\hbar)$，其中 $m_e$ 为电子质量，
$\mathbf v_A$ 为核速度，$\mathbf r$ 为电子坐标。在线性速度及原子局域近似下，
动能校正为 $\delta H_{IJ}=i\hbar\sum_Q\dot Q\,D^{IJ}:S_A^Q$。
它与电子传播方程的 $-i\hbar\sum_Q\dot Q\,d_{IJ}^Q$ 合并，得到

$$
\boxed{d_{IJ,\mathrm{ETF}}^Q=d_{IJ}^Q-d_{IJ,\mathrm{AO}}^Q
=\frac{h_{IJ}^Q}{\Delta\omega}+d_{IJ,\mathrm{orb}}^Q .}
$$

→ **解释：** 该校正不删除对称重叠/Pulay 项或额外的 MO 响应。
校正规则来自 [Fatehi–Subotnik 2012，式 (20)–(23)](https://doi.org/10.1021/jz3006173)
及 [Athavale 等 2023，式 (37)、(47)–(55)](https://arxiv.org/html/2308.14621v1#S3.SS1)。
对 NTTDA AWF 的应用来自上面的直接微分。

普通 CIS/SF-CIS 的粒子–粒子、空穴–空穴联络可使 ETF 恰好抵消整个 CSF 项，见
[Zhang–Herbert 2014，式 (16)–(28)](https://www.asc.ohio-state.edu/herbert.44/reprints/JCP_141_064104.pdf)。
NTTDA 的降低/保持自旋组态可以产生不同 C/O/V 子空间之间的反对称密度，
因此不能一般地丢弃 $d_{\mathrm{orb}}$。升自旋的 CC/VV 密度在上述规范下才使该项为零。

**代码对应：** `nac/nttda.py:compute_nac` 和 `grad/_nttda/frame.py:compute_frame`
均调用 `nac_csf_components`，ETF 选择 `components.orbital`，完整 NAC 选择
`components.total`；先乘能隙，再加入振幅分子。既有 GMRES 收敛阈值保持不变。

### 3. 回归检验

`gpu4pyscf/nac/tests/test_nttda_etf.py` 对每个位移重新收敛 SCF/NTTDA，
以自旋适配行列式重叠对齐根和相位。跨几何 AO 重叠
$B(s)=\langle\chi(Q-s)|\chi(Q+s)\rangle=S+2sS_A^Q+O(s^2)$，
用 $(B+B^T)/2$ 替换它只消去一阶 AO 联络，两侧 MO 系数和振幅仍然松弛。
由此构造的中心重叠差分与解析 ETF 比较；原始 $B$ 同时检查完整 NAC。
这个数值检验不调用解析 AO/轨道联络函数，其作用是验证前述规则的实现。

回归覆盖 HF/PBE、能隙缩放、非顺序原子选择及态交换；联合帧测试覆盖
四种方法入口、普通积分/DF、重复调用和已有的缓存失败处理。
固定网格 PBE 的差分通过不代表精确的平移不变性；积分及网格响应误差需另行处理。
本次定向验收结果及运行命令见 [ETF 修正验收](../nttda_refactor_validation.md#etf-修正验收2026-10-11)。
