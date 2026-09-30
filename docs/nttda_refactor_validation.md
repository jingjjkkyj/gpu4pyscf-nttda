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
