# NewCppRL

牧场覆盖 / 除草场景的自主导航 **C++/Python 混合强化学习研究项目**。

- **环境**:组件化 Gymnasium 环境(`envs_new/`,变体 v2/v4/v5/v6)+ pybind11 C++ 加速核(`cpu_apf` / `cpu_fov`)。
- **训练**:`rl_new/sac_cont_sy/` —— SAC 三阶段课程学习 + 分桶优先经验回放 + 异步评估;DQN 在 `rl_new/dqn/`。
- 架构细节见 `CLAUDE.md`。

---

## 环境部署

**已验证技术栈**:Python 3.12 · torch 2.12.0 · torchrl 0.12.0 · tensordict 0.12.4 · torchvision 0.27.0 · torchcodec 0.13.0 · numpy 2.4.6 · CUDA 13.x。

默认运行环境是 **`.venv`(新栈)**;`new_venv` 是升级前旧栈(tensordict 0.10.0)的**回退环境**。

> **为什么 torchrl 单独装**:torchrl 带需对齐 torch ABI 的 C++ 扩展(`_torchrl.so`,分桶优先回放的 SumSegmentTree)。PyPI 预编 wheel 可能与 torch 2.12 ABI 不匹配(`undefined symbol`),所以 **从源码对齐已装的 torch 现编**最稳。其余依赖全部由 `pip install -e .` 一步装好。

```bash
# 0) 建虚拟环境(Python 3.12)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip

# 1) 仅当目标服务器 CUDA ≠ 13.x 时:先按目标 CUDA 装 torch/torchvision。
#    本机 / CUDA 13.x 跳过本步 —— 第 2 步默认 PyPI wheel 即 cu130。
#    例(CUDA 12.4):
# pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu124

# 2) 装本项目 + 全部依赖,并编译 C++ 扩展 cpu_apf / cpu_fov
pip install -e .

# 3) 单独装 torchrl 0.12.0(源码,对齐已装的 torch 现编 _torchrl.so)
pip install 'pybind11[global]' ninja                # torchrl C++ 扩展构建依赖
git clone https://github.com/pytorch/rl.git torchrl_src
cd torchrl_src && git checkout 83f3d50215aa8841dc79a3189ec6f052feb52c60
pip install -e . --no-build-isolation               # --no-build-isolation:用当前环境已装的 torch 来编
python setup.py build_ext --inplace
cd ..

# 4) (仅当要加载升级前的旧 checkpoint 才需要)装旧 ckpt 反序列化 shim
#    旧 torchrl 用 CompositeSpec 等类名存的整模型 .pt,新 torchrl 已重命名 → 需别名桥接才能 torch.load。
SP=$(python -c "import site; print(site.getsitepackages()[0])")
cp deploy/legacy_ckpt_compat/_unpickle_compat.py "$SP/"
cp deploy/legacy_ckpt_compat/_unpickle_compat_loader.pth "$SP/"   # .pth 在解释器启动时自动 import shim

# 5) 自检
python -c "import torch, torchrl, tensordict, cpu_apf, cpu_fov; \
from torchrl._torchrl import SumSegmentTreeFp32; \
import torchrl.data.tensor_specs as s; assert hasattr(s,'CompositeSpec'); \
print('OK', torch.__version__, 'torchrl', torchrl.__version__, '| 旧ckpt shim OK')"
```

依赖钉版的唯一真源是 `pyproject.toml`(torchrl 除外,如上单独装)。旧的 `environment.yaml`(conda 导出)已过时、不再维护。

---

## 回退到升级前

`git checkout 2026-05-24`(升级前快照 tag)+ 改用 `new_venv`(旧栈 tensordict 0.10.0)即可完整回到 torch 2.12 升级前的状态。
