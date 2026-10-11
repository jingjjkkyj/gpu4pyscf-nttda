# GPU NTTDA 的 ΔS=0/+1 梯度和 NAC

## 推导与移植记录 — 2026-10-11

**目标：** 将 NEST 中的自旋保持、升自旋梯度与 NAC 接入 GPU 的现有
prepared derivative、J/K ledger、Z-vector 和联合帧流程。

**假设：** 共用实空间轨道、共线高自旋参考、Tamm–Dancoff 近似，
绝热 XC 核；DFT 导数不含积分网格响应。生产梯度目前沿用实轨道约定；
态间密度与行列式展开的代数测试另外包含复振幅。

**符号：** 闭壳层轨道为 $i,j\in C$，开壳层为 $u,v\in O$，空轨道为
$a,b\in V$；参考自旋为 $S=|O|/2$。$\Phi_{SM}$ 是归一化参考自旋分量，
$C_C,C_O,C_V$ 是 AO 到各 MO 子空间的系数。$F_0$ 是所选参考方法的公共
Fock 算符；$K_0$ 是包含精确交换的横向自旋响应核，$D_O=C_OC_O^T$。
冒号表示同序 AO 元素求和，不隐含矩阵乘法。

$$
F_z=\frac12 K_0[D_O].
$$

→ **解释：** $F_z$ 是 NTTDA 响应算符的一部分；不能用 ensemble 的 SCF
Fock 自旋差替代。杂化及长程交换均包含在 $K_0$ 内。

### 1. 升自旋求解器及梯度

升自旋最高权重组态只有闭壳层到空轨道的自旋翻转：

$$
|ia;S+1,S+1\rangle=-a_{a\alpha}^{\dagger}a_{i\beta}|\Phi_{SS}\rangle,
\qquad |\Psi_I\rangle=\sum_{ia}X^I_{ia}|ia;S+1,S+1\rangle.
$$

→ **解释：** 系数存为 `X[i,a]`；整体负号来自秩一张量的 $+1$ 分量。
这些组态正交归一，没有 $1/S$ 项，所以闭壳层 $S=0$ 也适用。

令 $F_V=C_V^T(F_0+F_z)C_V$、$F_C=C_C^T(F_0-F_z)C_C$，则

$$
(AX)_{ia}=\sum_bX_{ib}(F_V)_{ba}
-\sum_j(F_C)_{ij}X_{ja}
+\big[C_V^T K_0[T]C_C\big]_{ai},
\qquad T=C_VX^TC_C^T.
$$

→ **解释：** 粒子和空穴对 $F_0$ 的符号相反，对 $F_z$ 的符号相同；
最后一项是横向自旋核。`gen_vind_sfu` 按这个矩阵作用构造 Davidson 算符。

定义粒子、空穴 AO 密度及探针：

$$
D_p=C_VX^TXC_V^T,\qquad D_h=C_CXX^TC_C^T,
\qquad P_0=D_p-D_h,\qquad P_z=D_p+D_h.
$$

→ **解释：** 这正是 `spin_raising_fock_projections` 的 VV 和 CC 两项；
单态 $X$ 的平方必须在交叉极化时保留，不能重新归一化。

把矩阵作用与 $X$ 收缩，得到用于微分的标量：

$$
\mathcal G[X]=X^TAX=P_0:F_0+P_z:F_z+T:K_0[T].
$$

→ **解释：** 对 Fock 项、两个跃迁密度因子以及核的密度依赖分别求导，
得到轨道旋转 M 矩阵、固定 AO 直接项和 J/K ledger。

由于 $K_0$ 对密度内积自伴，固定核时有

$$
\delta\big(T:K_0[T]\big)=2\,\delta T:K_0[T],
\qquad
\delta T=\delta C_VX^TC_C^T+C_VX^T\delta C_C^T.
$$

→ **解释：** 这给出 `_hfx_projection_q` 中两个投影和因子 2；XC 的
跃迁因子、参考密度与核导数由原有 XC 后端负责。随后使用与 ΔS=−1 相同的
参考专属轨道 Hessian，而不是另建升自旋 CPHF。

闭壳层时 $D_O=F_z=0$，上述作用退化为标准闭壳层三重态 TDA。
`test_nttda_spin_raising.py` 用标准 PySCF 三重态 TDA 独立检查这个极限。

### 2. ΔS=0 的组态与相位

保留现有求解器顺序 `CO / CV / OO / OV / CV0`；其中 `OO` 是一个标量。
令 $T^{1,0}_{pq}=(a_{p\alpha}^{\dagger}a_{q\alpha}
-a_{p\beta}^{\dagger}a_{q\beta})/\sqrt2$，
$T^{1,+1}_{pq}=-a_{p\alpha}^{\dagger}a_{q\beta}$。
CO/CV/OV 的秩一组态先耦合再归一化：

$$
|pq;S,S\rangle\ \propto\
\sqrt{\frac{S}{S+1}}T^{1,0}_{pq}|\Phi_{SS}\rangle
-\frac{1}{\sqrt{S+1}}T^{1,+1}_{pq}|\Phi_{S,S-1}\rangle.
$$

→ **解释：** 两项是相同总自旋 $S$ 的最高权重耦合；逐组态归一化后与
求解器的欧氏振幅度量一致。此通道要求 $S\ge1/2$。

$$
|OO\rangle=|\Phi_{SS}\rangle,\qquad
|ia;CV0\rangle=-\frac{a_{a\alpha}^{\dagger}a_{i\alpha}
+a_{a\beta}^{\dagger}a_{i\beta}}{\sqrt2}|\Phi_{SS}\rangle.
$$

→ **解释：** CV0 的负号与现有 GPU 求解器及 NEST 一致；若在 AWF 中漏掉，
OO–CV0 的态间密度和 NAC 都会变号，但单个纯 CV0 组态的范数仍然正确。

对正交不同根，定义 $\gamma^{IJ}_{pq}=\langle\Psi_I|
\sum_\sigma a_{p\sigma}^{\dagger}a_{q\sigma}|\Psi_J\rangle$。
参考占据乘 $\langle\Psi_I|\Psi_J\rangle$ 的项为零。以最敏感的块为例：

$$
\gamma_{ia}^{IJ}=-\sqrt2\,(X_{OO}^{I})^*X_{ia,CV0}^{J},
\qquad
\gamma_{ai}^{IJ}=-\sqrt2\,(X_{ia,CV0}^{I})^*X_{OO}^{J}.
$$

→ **解释：** 用一体算符作用于上述组态即可得到负号和 $\sqrt2$。
其余块由相同的费米反对易代数得到，实现在 `interstate_rdm1_analytic`。
测试把所有块与独立行列式展开逐元素比较，同时检查最高权重条件和复共轭转置。

ΔS=+1 的态间密度仅有

$$
\gamma_{ji}^{IJ}=-\sum_a(X^I_{ia})^*X^J_{ja},\qquad
\gamma_{ab}^{IJ}=\sum_i(X^I_{ia})^*X^J_{ib}.
$$

→ **解释：** 升自旋只有空穴 CC 和粒子 VV 块，无不同占据子空间间的源项；
因此其额外 CSF 轨道响应为零，但 AO 联络一般不为零。

### 3. 交叉导数、Z 向量与 ETF

令 $g[X]$ 是仅激发能标量 $\mathcal G[X]$ 的全导数，则不同根之间

$$
h_{IJ}=X_I^TA^{[R]}X_J
=\frac14\left(g[X_I+X_J]-g[X_I-X_J]\right).
$$

→ **解释：** 这是实对称二次型的极化恒等式。不能混入参考能和核斥力梯度，
也不能归一化振幅之和/差。

`prepare_polarized_cross` 在求解前组合 M 矩阵、AO 直接项、Fock 探针和
尚未收缩的 J/K ledger。若参考轨道 Hessian 为 $H$、右端为 $b$，则

$$
H^Tz_\pm=b_\pm\quad\Longrightarrow\quad
H^T\frac{z_+-z_-}{4}=\frac{b_+-b_-}{4}.
$$

→ **解释：** 同一电子解的 Hessian 与振幅无关，因此 ΔS=0/+1 只需要一次
交叉 Z 向量求解和一次合并 J/K 导数收缩。二次准备仍计算两组振幅；
ΔS=−1 保留现有专用双线性准备。联合帧将 XC 响应批处理，并只计算一次交叉
$P_z$ 的 Fz 项。这个实现不声称与专用 −1 准备有相同的前处理开销。

最终令 $\Delta\omega=\omega_J-\omega_I$：

$$
d_{IJ}^{\mathrm{full}}=\frac{h_{IJ}}{\Delta\omega}
+d_{IJ}^{\mathrm{orb}}+d_{IJ}^{\mathrm{AO}},\qquad
 d_{IJ}^{\mathrm{ETF}}=\frac{h_{IJ}}{\Delta\omega}+d_{IJ}^{\mathrm{orb}}.
$$

→ **解释：** 三个通道使用同一个 ETF 定义；只删除 AO 联络。ΔS=+1 的
$d^{\mathrm{orb}}=0$ 是密度块结构的结果，不能推广到 ΔS=0/−1。
完整 ETF 论证和独立重叠差分见 [ETF 推导](nttda_etf.md)。

**代码对应：** `sftda/nttda.py:gen_vind_sfu`、
`grad/_nttda/delta_s_plus_one.py:prepare_grad_elec`、
`grad/_nttda/response.py:prepare_polarized_cross`、
`nac/nttda.py:interstate_rdm1_analytic`。
CPU 公式参照 NEST `src/nest/grad/nttda/delta_s_plus_one.py`、
`src/nest/nttda/nttda.py:transition_dm` 和 `src/nest/nac/_nttda_overlap.py`；
生产 GPU 代码不导入 NEST。

**代数验证：** `nac/tests/test_nttda_spin_channels.py` 直接检查行列式度量、
最高权重条件、复振幅态间密度及 MO 占据块重排。不另建重复验证脚本。

### 4. 严格本征残差下的子空间精度

闭壳层和双重态差分暴露出两种数值问题：预条件后的修正范数被核层能量差
压小；以及保留的 Ritz 基累积度量误差，在 $10^{-10}$ 的目标下停滞。
NTTDA 现在按对角能量跨度缩放线性相关阈值，并在目标小于双精度机器精度
平方根时启用完整子空间 QR。普通精度沿用原路径，公共 `eigh` 的新选项
默认关闭，不更改其它 TDDFT 调用的行为。

令 $B$ 的每行是一个试探向量，$Y=BA^T$ 是缓存的算符作用：

$$
B^T=QR,\qquad B'=Q^T,\qquad Y'=R^{-T}Y,\qquad
H'=(B')^*(Y')^T.
$$

→ **解释：** 同时变换试探基和缓存作用，保证表示的是同一个线性算符。
三角求解避免显式求逆，不增加 AO 响应调用或迭代上限，也不放宽收敛标准。
新的实数/复数近线性相关初猜回归用实际 $AX-\omega X$ 检查残差；
分子回归另外检查闭壳层 H2O 和双重态 NH2 的完整/ETF NAC 差分。
