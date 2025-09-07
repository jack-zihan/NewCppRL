# Repository Guidelines

## Project Structure & Module Organization
- `envs/` and `envs_new/`: Gymnasium-compatible mowing robot environments and factory/registration. Prefer `envs_new/` for new work.
- `rl/` and `rl_new/`: Reinforcement learning algorithms (TorchRL). Use `rl_new/` for SAC continuous control training/eval.
- `src/`: C++/pybind11 extension (`cpu_apf.cpp`) compiled to `cpu_apf*.so`.
- `configs/`: YAML configs for environments and training.
- `tests/` and `test_env_consistency/`: Pytest suites and analysis scripts; use for regression checks.
- `utils/`, `torchrl_utils*`: Helpers and shared utilities.
- Output folders (ignored): `ckpt/`, `logs/`, `outputs/`.

## Build, Test, and Development Commands
- Install with extension build: `pip install -e .` (builds `cpu_apf` via pybind11). Or: `python setup.py build_ext --inplace`.
- Run unit/integration tests: `pytest -q` (from repo root). Example quick smoke: `python tests/env_make.py`.
- Train (V4): `python rl_new/sac_cont/area_coverage_sac_cont_train.py collector.frames_per_batch=100 collector.total_frames=1000 collector.num_envs=1`.
- Evaluate (V5): `python rl_new/sac_cont/area_coverage_v5_sac_cont_eval.py --ckpt_path ckpt/area_coverage_v5_sac_cont/<run>/model.pt --episodes 5`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints where reasonable, docstrings for public APIs.
- Formatting: `black .`; Imports: `isort .`; Lint: `flake8` (fix before PRs).
- Naming: modules/packages `snake_case`; classes `CamelCase`; functions/vars `snake_case`.
- C++ (pybind11): follow existing style; keep headers minimal and functions small.

## Testing Guidelines
- Framework: `pytest`. Place tests in `tests/` or `test_env_consistency/` as `test_*.py`.
- Keep tests deterministic (set seeds); skip GPU-heavy paths unless essential.
- Cover: env reset/step, observation shapes/dtypes, reward/termination logic, and APF consistency.
- Run: `pytest -q`; for targeted files: `pytest -q tests/test_v4_v5_environments.py`.

## Commit & Pull Request Guidelines
- Commits: use Conventional Commits (e.g., `feat:`, `fix:`, `refactor:`). Keep messages imperative and scoped.
- PRs must include: clear description, reproduction steps, linked issues, test updates, and before/after metrics (logs, screenshots of maps if relevant).
- Don’ts: commit `ckpt/`, `logs/`, `*.mp4`, local envs (`venv/`, `.venv/`, `new_venv/`).

## Security & Configuration Tips
- Avoid hardcoded absolute paths; prefer config (`configs/*.yaml`) and `cfg.buffer.scratch_dir` for replay buffers.
- Do not embed secrets or tokens; use environment variables.
- Large artifacts belong outside the repo; keep results reproducible via configs and seeds.

## 优雅、高效、简洁、清晰代码设计理念

**核心哲学：Less is More - 用最简单的方式解决最复杂的问题**

追求业务本质与技术优雅的完美融合，但始终以实用主义为导向，绝不为了技术完美而牺牲简洁性和可理解性。

### 🚨 设计反面教材（必须避免的陷阱）
1. **过度抽象陷阱**：不要为了"架构完美"而创建复杂的抽象层。如果抽象不能显著减少代码或提高可理解性，就不要抽象，但如果确定能够提高代码简洁性和清晰性的抽象，可以不用过于保守，可以勇于信任自己的思考。。
2. **技术炫技心理**：不要为了展示技术能力而使用复杂的设计模式。用户关心的是功能，不是你的技术水平。
3. **过度工程化**：不要一开始就追求"完美的可扩展性"。先解决当前问题，再考虑未来扩展。
4. **API复杂化**：不要为了"功能完整"而创建复杂的API。简单易用的API胜过功能丰富的复杂API。

### 🔍 如何识别并消除过度工程化（实战经验）

#### 核心识别方法：三问法则
每当你设计或审查代码时，问自己三个问题：

1. **业务本质问题**："这个功能的本质需求是什么？"
   - 剥离所有技术实现，只看业务需要什么
   - 如果一句话说不清楚，说明理解还不够深入

2. **数据流向问题**："数据从哪里来，到哪里去？"
   - 追踪数据的完整路径
   - 如果中间有多次转换或存储，问"为什么需要这一步？"

3. **简化可能问题**："能否直接从A到B，而不经过C？"
   - 识别所有中间层
   - 挑战每个中间层的必要性

#### 实战案例：奖励系统优化全过程

**案例1：消除不必要的映射层**
```python
  # ❌ 过度工程化：多层映射
  class RewardSystem:
      COEFFICIENT_MAPPING = {
          'turn_gap': 'reward_turn_gap_coef',  # 第一层映射
          'turn_direction': 'reward_turn_direction_coef'
      }

      def _update_coefficients(self):
          for internal_name, config_name in self.COEFFICIENT_MAPPING.items():
              calc_name = self.CALC_MAPPING[internal_name]  # 第二层映射
              calc_class = self.CALCULATORS[calc_name]  # 第三层映射
              calc_class.coefficient = getattr(self.config, config_name)

  # ✅ 本质思考后：直接访问
  class RewardSystem:
      def calculate_reward(self):
          # 直接使用统一命名，无需映射
          coefficient = getattr(self.config, f"reward_{name}", 0.0)
  识别要点：当你发现自己在维护映射关系时，问"为什么不直接访问？"
  ```

案例2：消除不必要的状态存储
```
  # ❌ 过度工程化：重复存储
  class Calculator:
      coefficient = 0.0  # 类变量存储
  
      @classmethod
      def calculate(cls, env_state):
          return cls.coefficient * value  # 使用存储的值
  
  # 在RewardSystem中
  def _update_coefficients(self):
      Calculator.coefficient = self.config.coefficient  # 同步更新
  
  # ✅ 本质思考后：直接传递
  class Calculator:
      @classmethod
      def calculate(cls, env_state, coefficient):  # 作为参数传递
          return coefficient * value  # 直接使用参数
  识别要点：当你需要"同步"两处数据时，问"为什么要存两份？"
```
案例3：消除不必要的间接访问
```
  # ❌ 过度工程化：隐式传递+辅助方法
  def calculate(cls, env_state, coefficient, **kwargs):
      config = cls.get_config(kwargs)  # 辅助方法提取
      if not config:
          return 0.0

  @classmethod
  def get_config(cls, kwargs):
      return kwargs.get('config')  # 从kwargs提取

  # ✅ 本质思考后：显式参数
  def calculate(cls, env_state, coefficient, config=None):
      if not config:
          return 0.0  # 直接使用参数
  识别要点：当你创建"辅助方法"来访问数据时，问"为什么不直接传递？"
 ```

危险信号清单（Red Flags）

出现以下情况时，立即停下来重新思考：

1. 命名困难：想不出好名字，或名字很长很绕
  - 可能是抽象层次错误
2. 多层映射：A→B→C→D的转换链
  - 考虑直接A→D
3. 同步负担：需要保持多处数据一致
  - 使用单一数据源
4. 配置地狱：大量配置才能使用
  - 简化接口，提供合理默认值
5. 理解成本高：需要看多个文件才能理解一个功能
  - 减少抽象层次
6. 修改困难：简单需求需要改动多处
  - 重新组织代码结构

实践指南：逐步简化法

当你怀疑存在过度工程化时，按以下步骤操作：

1. 画出数据流图
  - 标记所有数据转换点
  - 识别冗余路径
2. 列出所有假设
  - "未来可能需要..."
  - "为了灵活性..."
  - 挑战每个假设的必要性
3. 尝试删除
  - 临时注释掉可疑的抽象层
  - 看是否能直接连接两端
  - 如果可以，永久删除
4. 重写对比
  - 用最简单的方式重写
  - 对比代码行数和复杂度
  - 选择更简单的版本

记住：优秀的设计让人感叹"原来这么简单"，而非"好复杂的架构"

### 🎯 核心设计原则（优先级排序）

#### 第一优先级：实用主义导向
1. **问题解决优先**：始终问"这能解决实际问题吗？"而不是"这个设计完美吗？"
2. **最小变更原则**：如果在现有代码基础上进行小幅改进即可完美解决问题，则优先小幅改进，当确实有足够的必要的时候也可以推倒重来
3. **5分钟理解测试**：任何设计如果不能在5分钟内被其他开发者理解，就需要简化

#### 第二优先级：简洁性保证
1. **代码行数约束**：解决问题的代码增量应该控制在合理范围
2. **文件数量控制**：优先修改现有文件，避免创建过多新文件增加认知负担
3. **依赖关系简化**：避免创建复杂的依赖关系图，保持线性或树状结构
4. **接口数量限制**：一个组件的公开接口不应超过7±2个（人类认知极限）
5. **最小化代码重复**：将真正公共的逻辑提取到合适的层次，但不为了消除表面的代码相似而创建不必要的抽象。
6. **过度工程化**：过度工程化的真正定义是一个可以简单高效实现使用了大量繁琐代码实现，使得更难理解和更难维护，但使用成熟、高效的库函数是明智的，不是过度工程化，往往可以用更少代码量取得更好的性能和清晰度。

#### 第三优先级：优雅性追求
1. **极简关注点分离**：每个类/方法只做一件事，但不要为了分离而分离
2. **自然适配差异**：通过合理的默认参数、可选参数等自然方式处理不同实现的差异，避免为了统一接口而强制传递无用参数或创建无意义的抽象层。
3. **语义驱动设计**：代码结构应反映业务本质，让代码意图一目了然
4. **渐进式演化支持**：支持平滑扩展，但不要为了未来可能性而过度设计
5. **状态一致性原则**：任何涉及状态变更的操作都要保证相关状态的同步更新，避免出现状态不一致的隐患。

### 🔧 实践判断标准

### 实践层面原则：
1. 业务域内聚原则：按功能域而非技术层组织代码结构，让代码结构反映业务理解而非技术实现，使开发者能用业务思维直接理解代码。
2. 完整生命周期管理：每个组件应负责完整的业务流程，避免功能碎片化和跨组件的复杂协调，一个组件解决一个完整问题。
3. 组合优于继承：通过组合小而专注的组件来构建复杂功能，而非深层继承体系。可插拔的架构设计让系统具备优雅的演化能力。
4. 精准信息传达：代码本身应清晰表达意图，当需要注释时，解释"为什么"而非"是什么"。
6. 接口一致性：相同类型的组件提供一致的接口和交互模式。通过统一的生命周期和调用约定，降低认知负担，提高系统可预测性。

#### 何时应该抽象？
✅ **应该抽象的情况**：
- 相同逻辑出现3次以上
- 抽象能显著减少代码量（>30%）
- 抽象能明显提高可读性
- 抽象的接口比原代码更简单

❌ **不应该抽象的情况**：
- 只是为了"消除重复"而抽象
- 抽象层比原代码更复杂
- 抽象只是为了"未来可能的需求"
- 抽象需要大量配置才能使用

#### 何时应该保持简单？
✅ **保持简单的信号**：
- 当前解决方案已经工作良好
- 问题域本身就很复杂，不需要额外的技术复杂性
- 团队成员都能轻松理解现有代码
- 改动影响面很小

❌ **过度复杂的信号**：
- 需要创建多个新的抽象概念
- 需要大量文档才能解释设计
- 其他开发者很难快速上手
- 解决小问题却引入大变更

### 💡 设计决策流程

每次设计决策时，按顺序问以下问题：

0. **本质识别**："剥离所有技术细节，这个功能到底在做什么？" [新增]
1. **必要性检查**："这个改动必要吗？..."
2. **简洁性评估**："最简单但优雅高效的解决方案是什么？"
3. **理解性测试**："其他人能在5分钟内理解这个设计吗？"
4. **影响面评估**："这个改动会影响多少现有代码？"
5. **维护性预测**："6个月后维护这段代码困难吗？"
6. **简化可能**："还能更简单吗？哪些是真正必要的？" [新增]

### 🎨 代码美学标准

#### 优秀代码的特征：
- **像散文一样流畅**：从上到下阅读时逻辑自然流畅
- **像诗歌一样精炼**：每一行代码都有存在的必然理由
- **像数学公式一样优雅**：简洁而富有表达力
- **像自然语言一样直观**：用业务思维就能理解

#### 警惕的代码气味：
- **过度设计气味**：为了"架构完美"而创建的复杂结构
- **技术炫技气味**：使用高深技术但不解决实际问题
- **过早优化气味**：为了"未来需求"而增加当前复杂性
- **抽象成瘾气味**：把简单问题包装成复杂抽象

### 🏆 终极目标

优秀的代码应该让人看完后感叹："原来可以这么简单！"而不是"这个设计真复杂！"
但是应该注意"简单” 指的是指的是可维护性、简洁性、效率和清晰性的综合考量，可以不用需要害怕过度工程化变得过于保守，过度工程化的真正定义：大量代码实现简单效果，而不是使用高效的工具， 使用成熟、高效的库函数是明智的，不是过度工程化。
**真实案例对比**：

```python
  # ❌ 让人皱眉的设计
  "这个RewardSystem为什么要三层映射？"
  "为什么coefficient要存在类变量里？"
  "get_config这个方法是干什么的？"

  # ✅ 让人赞叹的设计
  "哦，直接传参数就行了！"
  "原来奖励就是系数乘以变化量！"
  "代码和业务逻辑完全一致，真清晰！"

  这些添加内容基于我们的实战经验，提供了具体的识别方法、真实案例和实践指南，能帮助未来更好地避免过度工程
  化，真正实现"Less is More"的设计理念。
 ```

**成功的标志**：
- 用户说："这个设计很自然"
- 同事说："我一看就懂了"
- 自己说："维护起来很轻松"
- 回头看："当时为什么想得这么复杂？"

**失败的标志**：
- 需要大量文档解释设计
- 新人很难快速上手
- 简单需求需要复杂实现
- 经常需要重构核心架构

---

**记住：简洁是复杂的终极形式。真正的大师能用最简单的方式解决最复杂的问题。**

## 代码注释原则
核心理念：注释是为了增加代码的清晰性、可理解性、可维护性，解释"为什么"而非"是什么"
1. 不要给显而易见的操作添加注释（如简单的getter/setter）
2. 注释应该提高代码的清晰度和可维护性，而不是增加阅读负担
3. 只有复杂操作、运算流程复杂、核心功能才需要文字说明
4. 注释的目的是让开发人员更快理解代码逻辑

应该写注释的情况

- 复杂算法逻辑：数学公式、算法步骤、业务逻辑等需要解释计算目的和逻辑
- 设计决策说明：为什么选择某种实现方式、架构考虑、性能优化原因
- 非显而易见的关系：组件间的依赖关系、状态转换逻辑、边界条件处理
- 关键业务概念：领域特定的概念、复杂的数据结构含义

不应该写注释的情况

- 显而易见的内容：属性访问器、简单的getter/setter、明显的变量名
- 重复代码意图：方法名已经清晰表达的功能、参数名自解释的情况
- 僵硬的模板化注释：为了写注释而写的形式化文档，每个方法都套用相同模板

注释组织原则

- 类级别统一说明：常用参数、核心概念在类文档中统一解释，避免方法级重复
- 块级解释优于行级：对复杂逻辑进行分块解释，说明整体思路和关键步骤
- 精炼表达：用最少的文字传达最有价值的信息，每个注释都应有明确存在价值

判断标准：如果删除这个注释会让代码理解变困难，则保留；如果注释只是重述代 码内容，则删除。

## 注意
在每一次规划和代码优化行动前，思考目前需要解决什么问题，什么解决方案最好最合适，什么样的代码实现最优雅、高效、简洁、清晰，给出最好、最合适的方案，三思而后行，不要给不好的方法，增加人工矫正量。
