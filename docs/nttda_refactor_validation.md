# NTTDA grad/NAC 重构验收（2026-09-30）

## 交付内容

- GPU 的求解器、梯度、NAC 和 FSSH 使用本仓库公式；删除 forge loader、
  `nttda_bridge.py`、CPU twin、动态梯度子类和 `nttda_context.py`。
  显式调用 ensemble `to_cpu()` 仍需要 forge；正常 GPU 计算不经过此接口。
- 两端的方法定义均合并为 `sftda/nttda_methods.py`。方法记录只包含身份，
  不保存算法回调。`PreparedGradient` 显式保存 M、直接项、探针和 J/K 收缩表。
- ROKS/ensemble 轨道方程合并为 `orbital.py`；NoBeta 修正并入 XC 模块。
  CPU 的梯度及方法定义文件由 17 个减为 10 个。CPU 的 GPU 注入接口已删除。
- GPU 公共梯度类为静态类，`base` 保持原 GPU TD 对象。
  `compute_frame` 保持公开导入路径，调度实现放在 `_nttda/frame.py`。
- 默认采用历史已验证的 AO 归约、fused XC、按输出槽归并的 DF、精确交换因子、
  Fock 缓存和 selected-reference RHS 合并；移除非槽位 `rank_batched` 实验分支。
- 修复移植边界问题：真实长程 J、host/device 密度缓存键、EnsembleROKS HF 响应、
  位移 ROKS 的泛函/DF 配置、NAC scanner 缓存更新和 FSSH 检查点数组转换。
- 公共构造器、状态编号、ETF/gap 约定及严格 GMRES 真实残差标准保持不变。

## 本地验证

环境：RTX 4060 Laptop 8 GiB、CUDA 12.4、PySCF 2.8、NumPy 2、CuPy；
BLAS/OpenMP 线程数为 2。cuTENSOR 加载失败后使用现有 CuPy contraction 路径。
未修改任何数值验收阈值。

| 检查 | 结果 |
|---|---|
| CPU 方法、梯度分层、标量公式、轨道响应、参考梯度、有限差分和 NAC | 55 passed，135 subtests passed |
| CPU 禁止导入 GPU/CuPy 的独立进程 | 1 passed |
| GPU 核心：四方法对拍、独立安装、联合帧、缓存、DF 有限差分、FSSH 和算符优化 | 153 passed，13 subtests passed |
| GPU 补充：NAC scanner、AO/fused XC、DF 核、参考梯度与求解器 | 54 passed；1 个原有 DF 指纹测试失败，见下文 |
| 静态检查 | 两仓库 `git diff --check`、新增核心模块 F821/F822 检查通过 |

CPU/GPU 对拍固定同一轨道、网格、振幅、根序和相位，覆盖四种方法、
`deltaS=-1,0`、HF/LDA/PBE/TPSS/CAM-B3LYP。梯度阈值为 `1e-7`，
完整 NAC 阈值为 `1e-6`。GPU 独立进程禁止导入 `pyscf.sftda`、
`pyscf.grad.nttda` 和 `pyscf.nac`，实际运行求解器、梯度、完整 NAC 和联合帧。

与 CPU 重构前 `db960abb` 保存的 HF/PBE 结果比较：

| 输出 | 数组数 | 最大绝对差 |
|---|---:|---:|
| 激发能 | 16 | 2.70e-14 |
| 总梯度 | 16 | 8.10e-14 |
| 完整 NAC（对齐整体相位） | 8 | 4.18e-12 |

核心测试入口：

- CPU：`pyscf/sftda/test/test_nttda_methods.py`、`pyscf/grad/test/test_nttda_*.py`、
  `pyscf/nac/test/test_nttda.py`、`test_nttda_ensemble.py`。
- GPU：`gpu4pyscf/grad/tests/test_nttda_native.py`、`test_nttda_methods.py`、
  `test_nttda_adapter.py`、`test_nttda_params.py`、`test_nttda_operator_optimizations.py`、
  `test_ensemble_roks_nttda_df.py`、`gpu4pyscf/fssh/tests/test_fssh_nttda.py`。
- 补充 DF：`gpu4pyscf/df/tests/test_df_tdrhf_grad.py`，包括保留的
  slot-grouped 与普通输出逐项对拍。

## 已知限制

`test_jk_energy_per_atom` 的固定数值指纹相差 `1.865e-9`，超过其 `1e-9` 阈值。
本地使用原始 GPU HEAD `a25f8159` 的 DF 实现复测同一项，也以相同的
`1.865e-9` 差异失败；新旧实际指纹只差约 `2.84e-14`，不是本次重构引入。
结果记录在 `/tmp/nttda-refactor/gpu-baseline-fingerprint.log`。没有放宽该测试阈值。

本次没有重新测量 A100 大体系性能，不能把历史加速比当成本次重构的实测结果。
LDA、NoBeta MGGA 及部分独立 post-Z XC 保留本仓库中的标准 PySCF NumInt
host quadrature；这不需要 forge。大型体系和 cuTENSOR 后端的性能验收仍需补做。

本次运行日志保存在 `/tmp/nttda-refactor/`：`cpu-acceptance.log`、
`cpu-independent.log`、`gpu-acceptance.log`、`gpu-kernels.log`，
以及 `baseline_comparison.json`。上述测试在提交源码前完成。

## ETF 修正验收（2026-10-11）

独立 NAC 和 `compute_frame` 的 `use_etfs=True` 现在只删除反对称 AO 联络，
保留移动组态基的额外轨道响应。完整 NAC 的公式保持不变。
这修正了上述历史版本的 ETF 行为；当前公式见 [推导](derivations/nttda_etf.md)。

新增独立重叠差分回归在修改前先复现 HF 算例的 ETF 最大误差
`0.04729087` Bohr^-1，同时完整 NAC 检查通过。修正后 HF/PBE 的完整/ETF
差分均满足原定 `3e-5` Bohr^-1 阈值，没有放宽收敛或验收条件。

环境：WSL CUDA GPU、Python 3.10、PySCF 2.8.0、CuPy 13.4.1；
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`，
`PYTHONPATH=.`。本地环境使用
`LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/jing/anaconda3/envs/gpu4pyscf_dev/lib`。

| 检查 | 结果 |
| --- | --- |
| HF/PBE 独立 ETF 差分、能隙/原子顺序/态交换 | 4 passed |
| 四方法、普通/DF 联合帧、重复调用及缓存失败处理 | 8 passed |
| 完整/ETF scanner、DF 多梯度联合帧、DF 完整 NAC 差分、FSSH | 34 passed，13 subtests passed |
| 修改的 Python 文件 Ruff、`git diff --check` | 通过 |

在上述环境中运行以下定向命令；没有运行全库测试：

```bash
python -m pytest gpu4pyscf/nac/tests/test_nttda_etf.py \
  gpu4pyscf/grad/tests/test_nttda_methods.py -k 'etf or joint_frame' -q
python -m pytest \
  gpu4pyscf/grad/tests/test_nttda_native.py::test_nac_scanner_uses_the_new_electronic_solution \
  gpu4pyscf/grad/tests/test_ensemble_roks_nttda_df.py::test_compute_frame_batches_multiple_gradients_and_nac \
  gpu4pyscf/grad/tests/test_ensemble_roks_nttda_df.py::test_df_nac_wavefunction_difference_and_joint_frame \
  gpu4pyscf/fssh/tests/test_fssh_nttda.py -q
```

ETF 现在也执行已有的 CSF 轨道响应求解，计算量相应增加；本次未测量大型体系性能。
DFT 仍采用不含网格响应的导数模型。本次修正未扩展 GPU 的自旋通道支持范围。

## ΔS=0/+1 移植验收（2026-10-11）

在 ETF 修正 `9dd1ec7` 之后，增加原生升自旋求解器、升自旋梯度、两种新通道的
完整/ETF NAC、态间密度、AWF、scanner 和联合帧。三种通道共用已有参考方法、
Z 向量和 J/K ledger；没有新增生产 NEST 或 forge 依赖。
公式及代码对应见 [自旋通道推导](derivations/nttda_spin_channels.md)。

同一 WSL CUDA 环境，使用标准 PySCF 2.8.0 / CuPy 13.4.1 / Python 3.10。
NEST 对拍在另一个 Python 环境中运行，PySCF 2.13.1 / NumPy 2.2.6，
通过 NPZ 传递同一组 MO、占据、网格、振幅、能量和相位，不混合两套包的导入。

### 独立验证与范围

- HF/PBE/CAM-B3LYP 的闭壳层 +1 激发能与标准 PySCF 三重态 TDA 一致，
  阈值 `1e-8` Eh。对拍重建最终密度的 Fock 对角，避免把 SCF 上一步轨道能
  与 NTTDA 当前 Fock 混作同一算符。
- 随机复振幅覆盖 S=0、1/2、1、3/2 的适用通道，检查 AWF 范数、最高权重条件、
  MO 占据块重排和态间密度；解析密度与行列式展开阈值 `1e-13`。
- 15 组分子参考覆盖 ROKS、NoBeta、EnsembleRKS、EnsembleROKS、普通/DF、
  HF/PBE/CAM-B3LYP/TPSS，以及闭壳层 H2O +1 和双重态 NH2 0。
  总梯度与能量差分，完整/ETF NAC 与独立 AWF 重叠差分，均满足 `3e-5`。
- 新通道的梯度差分步长为 `1e-4` Bohr，NAC 为 `5e-5` Bohr。
  CH2 +1 的 `1e-4` 二阶差分截断误差约 `4e-5`，步长减半后降到验收范围；
  同时检查 `2e-4` 和 `5e-4`，观察到相应的二阶步长行为。
- 四种方法、三个通道、普通/DF 的独立与联合帧、重复调用和缓存失败处理；
  B3LYP/TPSS 的多梯度与 NAC 联合帧；三个通道的完整/ETF scanner 和禁止
  导入 forge 的独立进程均通过。原有 −1/FSSH 回归通过。
- 相位错误的进程内变异检查：反转 CV0 相位后，随机态间密度测试明确失败，
  最大误差约 `0.37`；测试不是只比对两条相同实现。

### NEST 对拍

ROKS/NoBeta × ΔS=0/+1 × HF/LDA/PBE/TPSS/CAM-B3LYP，共 20 组。
能量使用两边独立求解结果；导数使用固定同一组振幅，消除任意根相位差。

| 量 | 最大绝对差 |
| --- | ---: |
| 激发能 / Eh | 3.95e-12 |
| 总梯度 / Eh Bohr^-1 | 5.91e-13 |
| 完整 NAC / Bohr^-1 | 4.70e-12 |
| ETF NAC / Bohr^-1 | 4.72e-12 |

### 收敛与数值对拍标准

闭壳层和双重态的严收敛回归发现 Davidson 的预条件修正筛选及子空间度量问题。
现在按对角能量跨度缩放线性相关阈值，NTTDA 在小于机器精度平方根的目标下
启用可选完整子空间 QR。实际残差和原迭代上限保持不变；共享 `eigh` 的该选项
默认关闭。实/复近线性相关初猜都以实际算符残差验证，阈值为 `1e-10`。

交叉准备与两次完整梯度极化的比较：普通积分使用 `1e-10`，DF 使用已有联合帧
测试的 `5e-9`。新测试最初对 DF 也指定 `1e-10`，两个 CAM-B3LYP/DF 案例分别
相差约 `1.90e-10` 和 `9.57e-10`；将伴随容差从 `1e-12` 收紧到 `1e-14`，
或使用保留的 `legacy` DF 收缩路径，差异都未明显变化。因此采用已有 DF
一致性标准，不把新内部对拍设为比既有 DF 验收更严。独立物理差分与 NEST
对拍的验收标准没有放宽。

主批次为 `174 passed, 2 failed, 13 subtests passed`，上述两个失败仅来自
新 DF 内部对拍容差。修正该测试标准后，全部 15 个交叉准备/态交换测试重跑
通过。最终覆盖 **176 个不同测试、13 个子测试**；修改文件的 Ruff 和
`git diff --check` 通过。没有运行全库测试。

定向命令（沿用上一节的环境变量）：

```bash
python -m pytest \
  gpu4pyscf/sftda/tests/test_nttda.py \
  gpu4pyscf/sftda/tests/test_nttda_spin_raising.py \
  gpu4pyscf/nac/tests/test_nttda_spin_channels.py \
  gpu4pyscf/nac/tests/test_nttda_etf.py \
  gpu4pyscf/grad/tests/test_nttda_methods.py \
  gpu4pyscf/grad/tests/test_nttda_native.py::test_nac_scanner_uses_the_new_electronic_solution \
  gpu4pyscf/grad/tests/test_nttda_native.py::test_solver_gradient_and_nac_do_not_import_forge \
  gpu4pyscf/grad/tests/test_ensemble_roks_nttda_df.py::test_compute_frame_batches_multiple_gradients_and_nac \
  gpu4pyscf/fssh/tests/test_fssh_nttda.py -q
python -m pytest gpu4pyscf/nac/tests/test_nttda_spin_channels.py -k cross_preparation -q
```

本机日志：`/tmp/nttda-spin-final.log`、`/tmp/nttda-spin-cross-final.log`、
`/tmp/nttda-spin-cpu-final.log`；对拍数据位于 `/tmp/nttda-spin-channels/`。

### 限制

NAC 连接同一自旋通道中的不同根；没有实现跨通道自旋轨道耦合。
FSSH 适配器仍限定 ΔS=−1。ΔS=0 的近似根跟踪请求使用精确 AWF 重叠，
大型 AWF 的成本更高。严收敛 QR 和新通道的大体系 GPU 性能尚未测量。
DFT 仍采用不含网格响应的导数模型。
