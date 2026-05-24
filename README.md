# NewCppRL

牧场覆盖 / 除草场景的自主导航 **C++/Python 混合强化学习研究项目**。

- **环境**:组件化 Gymnasium 环境(`envs_new/`,变体 v2/v4/v5/v6)+ pybind11 C++ 加速核(`cpu_apf` / `cpu_fov`)。
- **训练**:`rl_new/sac_cont_sy/` —— SAC 三阶段课程学习 + 分桶优先经验回放 + 异步评估;DQN 在 `rl_new/dqn/`。
- 架构细节见 `CLAUDE.md`。

---

## 环境部署

**已验证技术栈**(本机 CUDA 13.2 / RTX 4090):Python 3.12 · torch 2.12.0+cu130 · torchrl 0.12.0 · tensordict 0.12.4 · torchvision 0.27.0 · torchcodec 0.13.0 · numpy 2.4.6。

默认运行环境是 **`.venv`(新栈)**;`new_venv` 是升级前旧栈(tensordict 0.10.0)的**回退环境**。

部署 = **两步**:① `pip install -e .` 装全栈 + 编 `cpu_apf`/`cpu_fov`;② **torchrl 单独源码装** —— 它的 `_torchrl.so`(分桶优先回放用的 SumSegmentTree)必须对齐 torch ABI 现编,PyPI 预编 wheel 可能 `undefined symbol`。

```bash
# 0) 建虚拟环境(Python 3.12)
python3.12 -m venv .venv && source .venv/bin/activate && pip install -U pip

# 1) 装本项目 + 全部依赖(torch 默认即 cu130 wheel)+ 编译 C++ 扩展 cpu_apf / cpu_fov
pip install -e .

# 2) torchrl 0.12.0 源码装(对齐已装的 torch 现编 _torchrl.so)
pip install 'pybind11[global]'            # 必需(torchrl 构建检查);可选 ninja 装了编更快
git clone https://github.com/pytorch/rl.git torchrl_src
cd torchrl_src && git checkout v0.12.0     # v0.12.0 = 当前最新 release,也正是本项目验证用的版本
pip install -e . --no-build-isolation     # --no-build-isolation:用当前环境已装的 torch 来编
python setup.py build_ext --inplace
cd ..

# 3) (可选,仅当要加载升级前的旧 checkpoint 才需要)旧 ckpt 反序列化 shim
#    旧 torchrl 用 CompositeSpec 等类名存的整模型 .pt,新 torchrl 已重命名 → 需别名桥接才能 torch.load。
SP=$(python -c "import site; print(site.getsitepackages()[0])")
cp deploy/legacy_ckpt_compat/_unpickle_compat.py "$SP/"
cp deploy/legacy_ckpt_compat/_unpickle_compat_loader.pth "$SP/"   # .pth 在解释器启动时自动 import shim

# 4) 自检
python -c "import torch, torchrl, tensordict, cpu_apf, cpu_fov; \
from torchrl._torchrl import SumSegmentTreeFp32; \
import torchrl.data.tensor_specs as s; assert hasattr(s,'CompositeSpec'); \
print('OK', torch.__version__, 'torchrl', torchrl.__version__, '| 旧ckpt shim OK')"
```

依赖钉版的唯一真源是 `pyproject.toml`(torchrl 除外,如上单独装)。旧的 `environment.yaml`(conda 导出)已过时、不再维护。

### CUDA 12 服务器(**不自动兜底**,需 2 处手动覆盖)

钉版默认是 **CUDA 13**(`torch` 默认 cu130 wheel + `cupy-cuda13x`)。目标机是 CUDA 12 时**不会自动降级**,要手动改两处:

1. **torch 只有 cu126 变体可用**(torch 2.12 在 `cu124`/`cu128` 上**不存在**,只有 `cu130`/`cu126`)→ 在第 1 步**之前**先按 cu126 装好,pip 就不会再去拉 cu130:
   ```bash
   pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
   ```
2. **cupy 换 12x**:`pip install -e .` 会装 pyproject 里的 `cupy-cuda13x`(CUDA 12 上加载会失败)→ 装完换掉(或先把 pyproject 那行改成 `cupy-cuda12x`):
   ```bash
   pip uninstall -y cupy-cuda13x && pip install cupy-cuda12x==14.1.0
   ```

要求目标机 NVIDIA 驱动支持 **CUDA ≥ 12.6**;更老驱动跑不了 torch 2.12(需更低版本 torch,超出本栈)。**注:CUDA 12 路径仅确认了 wheel 存在,未在真实 CUDA 12 机器上实测。**

---

## 回退到升级前

`git checkout 2026-05-24`(升级前快照 tag)+ 改用 `new_venv`(旧栈 tensordict 0.10.0)即可完整回到 torch 2.12 升级前的状态。
