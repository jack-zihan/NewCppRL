# NewCppRL

牧场覆盖 / 除草场景的自主导航 **C++/Python 混合强化学习研究项目**。

- **环境**:组件化 Gymnasium 环境(`envs_new/`,变体 v2/v4/v5/v6)+ pybind11 C++ 加速核(`cpu_apf` / `cpu_fov`)。
- **训练**:`rl_new/sac_cont_sy/` —— SAC 三阶段课程学习 + 分桶优先经验回放 + 异步评估;DQN 在 `rl_new/dqn/`。
- 架构细节见 `CLAUDE.md`。

---

## 环境部署(uv 管理,默认)

**已验证技术栈**(本机 CUDA 13.2 / RTX 4090):**Python 3.14.5** · torch 2.12.0+cu130 · torchrl 0.12.0 · tensordict 0.12.4 · torchvision 0.27.0 · torchcodec 0.13.0 · numpy 2.4.6 · dubins。

默认环境 = **uv 管理的 `.venv`(Python 3.14)**;`new_venv` 是更早旧栈(tensordict 0.10.0)的回退环境。

依赖唯一真源 = `pyproject.toml` 的 `[tool.uv]` + **`uv.lock`**(torch 走 `[tool.uv.index]` 的 cu130;dubins 在 `uv sync` 时由 `[tool.uv.extra-build-dependencies]` 注入 cython 现场重生成 `.c`)。**只有 torchrl 例外**:它的 `_torchrl.so` 必须对齐已装 torch 的 ABI 现编,而 `uv lock` 在 metadata 阶段拿不到 torch(鸡生蛋),故 torchrl 是 `uv sync` 之后的源码步骤(下面第 2 步)。

```bash
# 0) 装 uv(若没有)+ Python 3.14.5
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.14.5

# 1) uv sync —— 按 uv.lock 装全栈(torch cu130 / tensordict / cupy-cuda13x / ...)
#    + 现场 cython 化并编 dubins + 编本项目 C++ 扩展 cpu_apf / cpu_fov
uv sync

# 2) torchrl 0.12.0 源码装(对齐已装 torch 现编 _torchrl.so;不进 uv.lock,见上)
uv pip install 'pybind11[global]'
git clone https://github.com/pytorch/rl.git torchrl_src
cd torchrl_src && git checkout 83f3d50215aa8841dc79a3189ec6f052feb52c60   # =0.12.0+g83f3d5021,本项目验证用的源
uv pip install -e . --no-build-isolation && python setup.py build_ext --inplace && cd ..

# 3) (可选,仅加载升级前旧 checkpoint 才需)旧 ckpt 反序列化 shim
SP=$(uv run python -c "import site; print(site.getsitepackages()[0])")
cp deploy/legacy_ckpt_compat/_unpickle_compat.py "$SP/"
cp deploy/legacy_ckpt_compat/_unpickle_compat_loader.pth "$SP/"

# 4) 自检
uv run python -c "import torch,torchrl,tensordict,dubins,cpu_apf,cpu_fov; \
from torchrl._torchrl import SumSegmentTreeFp32; \
import torchrl.data.tensor_specs as s; assert s.CompositeSpec is s.Composite; \
print('OK', torch.__version__, 'torchrl', torchrl.__version__, '| dubins+shim OK')"
```

旧的 `pip install -e .` 流程已被 uv 取代;`environment.yaml`(conda 老导出)早已过时、不再维护。

### CUDA 12 服务器(**不自动兜底**,需 2 处手动覆盖)
默认是 CUDA 13(`[[tool.uv.index]]` 指 cu130 + 依赖 `cupy-cuda13x`)。CUDA 12 机器要手动改 `pyproject.toml` 两处后再 `uv lock && uv sync`:
1. `[[tool.uv.index]]` 的 url → `https://download.pytorch.org/whl/cu126`(torch 2.12 只有 cu130/cu126,**没有** cu124/cu128)。
2. 依赖 `cupy-cuda13x` → `cupy-cuda12x`。

要求目标机驱动支持 **CUDA ≥ 12.6**。**注:CUDA 12 路径仅确认 wheel 存在,未在真实 CUDA 12 机器上实测。**

---

## 回退(三档)
- **回退到 pip + Python 3.12 栈**(uv 切换前):`git checkout f4b1a6338`(那次提交的 `pyproject` 是 pip 版)→ `python3.12 -m venv .venv && .venv/bin/pip install -e .` + 上面第 2 步的 torchrl 源码安装。
- **回退到升级前**(torch 2.8 / tensordict 0.10):`git checkout 2026-05-24` + 改用 `new_venv`。
- torchrl 的 editable 源在 `/home/lzh/rl_remote_uv314`(当前 `.venv` 依赖它,**勿删**)。
