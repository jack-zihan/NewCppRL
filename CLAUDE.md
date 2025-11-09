# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
运行环境是 new_venv/bin/activate, 而不是venv/bin/activate

## Repository Overview

This is a hybrid C++/Python reinforcement learning research project for autonomous robotic navigation in pasture environments. The system combines high-performance simulation with state-of-the-art deep RL algorithms, specifically designed for agricultural robotics applications.

### 🚀 Architectural Migration

The project has undergone a fundamental transformation to support advanced research capabilities:

- **Environment System**: `envs/` → `envs_new/`
  - From monolithic design to component-based architecture with dependency injection
  - Dynamic composition enables runtime environment modification

- **Training System**: `rl/` → `rl_new/sac_cont_sy/`
  - From basic RL to three-stage curriculum learning pipeline
  - Sophisticated experience replay and evaluation systems

### 🎯 Core Innovations

1. **Component-Based Environment Architecture**
   - Dynamic component composition with dependency injection
   - Runtime environment modification without code changes
   - Four specialized environment variants (v2/v4/v5/v6)

2. **Three-Stage Curriculum Learning**
   - Automatic progression: S1 (exploration) → S2 (efficiency) → S3 (optimization)
   - Performance-based stage transitions with stability checks
   - Runtime reward shaping and sampling strategy adaptation

3. **Bucketed Prioritized Experience Replay**
   - Three-bucket classification: SUCCESS/NEAR_END/MID
   - Dynamic sampling ratios adjusted per curriculum stage
   - Fallback mechanism ensures training stability

4. **Asynchronous Evaluation Pipeline**
   - Non-blocking parallel evaluation with ThreadPoolExecutor
   - Sequential result ordering for deterministic curriculum decisions
   - Maintains training throughput while gathering metrics

### 🛠️ Technology Stack

- **RL Framework**: TorchRL with TensorDict for efficient data handling
- **Algorithm**: Soft Actor-Critic (SAC) with continuous action space
- **Environment**: Gymnasium-compatible with C++ acceleration via pybind11
- **Optimization**: CUDA graphs, mixed precision training, torch.compile support
- **Configuration**: Hydra for flexible experiment management

## Key Architecture Components

### 🌍 Environment System (`envs_new/`)

#### Component-Based Architecture Overview

The environment system follows a component-based design with dependency injection, enabling dynamic composition and runtime modification:

```
CppEnvBase (Orchestrator)
    │
    ├── Shared EnvironmentConfig ──→ All components reference single config instance
    │
    ├── ScenarioGenerator
    │   ├── MapCreator components (obstacle, field, weed, coverage, hif)
    │   └── Generates: agent, maps_dict, env_state
    │
    ├── EnvironmentDynamics
    │   ├── Updater components with dependency resolution
    │   ├── Topological sort ensures correct execution order
    │   └── Dynamic add_updater/remove_updater support
    │
    ├── ObservationGenerator
    │   ├── Multi-channel observation stacking
    │   └── Specialized variants (OrientationAwareObservationGenerator for v5)
    │
    ├── RewardSystem
    │   ├── Calculator components with group coefficients
    │   └── Runtime coefficient updates for curriculum learning
    │
    └── ActionProcessor
        └── Action space handling (discrete/continuous/multi-discrete)
```

#### Core Design Patterns

##### 1. StateVariable Pattern - History Tracking with Automatic Deltas
```python
class StateVariable[T]:
    """Generic state variable with configurable history tracking"""
    def __init__(self, name: str, history_length: int = 2, initial_value: T = None):
        self._history: deque[T] = deque(maxlen=history_length)

    @property
    def current(self) -> Optional[T]:
        return self._history[-1] if self._history else None

    @property
    def previous(self) -> Optional[T]:
        return self._history[-2] if len(self._history) >= 2 else None

    def change(self, steps_back: int = 1) -> Any:
        """Automatic delta computation for numeric/tuple types"""
        if isinstance(current, (int, float)):
            return current - past
        elif isinstance(current, tuple):
            return tuple(c - p for c, p in zip(current, past))
```

##### 2. Updater Pattern - Dynamic Components with Dependencies
```python
class Updater:
    """Base class for environment state updaters"""
    @classmethod
    def get_dependencies(cls) -> List[str]:
        """Declare dependencies for topological sorting"""
        return []  # Override in subclasses

    def update(self, state: Dict[str, Any]) -> None:
        """Perform state updates - implement in subclasses"""
        pass

# Automatic dependency resolution in EnvironmentDynamics:
sorted_updaters = topological_sort(updaters, dependencies)
for updater in sorted_updaters:
    updater.update(state_dict)
```

##### 3. Calculator Pattern - Composable Rewards with Groups
```python
class RewardCalculator:
    """Base class for reward components"""
    group = None  # Optional: "field_group", "turning_group" for coefficient multiplication

    @classmethod
    def calculate(cls, env_state, coefficient: float, config, **kwargs) -> float:
        """Calculate reward component - coefficient includes group multiplication"""
        return coefficient * computed_value

# Automatic coefficient resolution in RewardSystem:
coefficient = getattr(config, f"reward_{name}", 0.0)
if calculator.group:
    group_coef = getattr(config, f"reward_{calculator.group}_coef", 1.0)
    coefficient *= group_coef
```

#### Environment Versions Comparison

| Version | Task | Key Innovation | Observation Channels | Core Components |
|---------|------|----------------|---------------------|-----------------|
| **v2** | Weeding in unknown environment | APF (Artificial Potential Field) for weed attraction | 4: field, obstacle, weed, trajectory | `APFCalculator` with GPU/CPU adaptive computation<br>`WeedRemovalCalculator` for sparse rewards |
| **v4** | Pure field coverage | Overlap tracking without weeding | 3: field, obstacle, trajectory | `FieldCoverageUpdater` replaces exploration<br>`CoverageMapCreator` for overlap counting |
| **v5** | HIF-guided coverage | Human Intention Field with orientation | 5: field, obstacle, trajectory, global_cosine, global_sine | `OrientationAwareObservationGenerator`<br>Double-angle encoding for axial symmetry |
| **v6** | Spatiotemporal coverage | Decaying reward fields over time | (Configuration similar to v5) | Temporal reward decay mechanisms |

#### Detailed Environment Implementations

##### v2: APF-Enhanced Weeding Environment
- **File**: `envs_new/cpp_env_v2.py`
- **Purpose**: Learn exploration strategies with sparse weed rewards
- **Key Innovation**: Artificial Potential Field
  ```python
  # Exponential decay potential field
  def get_discounted_apf(binary_map, propagate_distance):
      distance_map = gpu_apf_bool(binary_map) if cuda else cpu_apf_bool(binary_map)
      gamma = (propagate_distance - 1) / propagate_distance
      potential_field = gamma ** distance_map  # Exponential decay
      return potential_field
  ```
- **Reward Components**: Base penalty + Weed removal + APF guidance + Field exploration

##### v4: Pure Coverage Environment
- **File**: `envs_new/cpp_env_v4.py`
- **Purpose**: Maximize coverage while minimizing overlap
- **Key Changes from Base**:
  ```python
  # Remove weed-related components
  self.scenario_generator.remove_component('weed')
  self.reward_system.remove_calculator('weed_removal')
  self.env_dynamics.remove_updater('weed')

  # Replace with coverage-specific components
  self.env_dynamics.remove_updater('field')
  self.env_dynamics.add_updater('field', FieldCoverageUpdater())
  self.scenario_generator.add_component('coverage', CoverageMapCreator())
  self.env_dynamics.add_updater('coverage_overlap', CoverageOverlapUpdater())
  ```
- **Default Configuration**: No obstacles, 768×768 resolution, trajectory enabled

##### v5: HIF-Guided Coverage Environment
- **File**: `envs_new/cpp_env_v5.py`
- **Purpose**: Human-intention-guided coverage with orientation awareness
- **Key Innovation**: Double-angle encoding for orientation
  ```python
  # Double-angle encoding ensures θ and θ+180° map to same vector
  double_angles = 2.0 * angles_image
  global_cosine = np.cos(double_angles) * confidence
  global_sine = np.sin(double_angles) * confidence
  ```
- **HIF Construction**: Frontier-based diffusion with exponential decay
- **Configuration Parameters**: `hif_propagate_distance`, `hif_decay_gamma`

#### Extension Guide

##### Adding a New Environment Version
```python
class CppEnvV7(CppEnvBase):
    def __init__(self, **kwargs):
        # Set version-specific defaults
        v7_defaults = {
            'use_your_feature': True,
            'your_parameter': 100
        }
        super().__init__(**{**v7_defaults, **kwargs})

        # Modify components
        self.scenario_generator.add_component('your_map', YourMapCreator())
        self.env_dynamics.add_updater('your_updater', YourUpdater())
        self.reward_system.add_calculator('your_reward', YourCalculator())

    def _get_observation_channels(self) -> int:
        return 4  # Your channel count

    def _get_observation_maps(self) -> Dict:
        # Define observation layers
        return {'field': {...}, 'obstacle': {...}, 'your_layer': {...}}
```

##### Adding a New Updater Component
```python
class YourUpdater(Updater):
    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['agent_motion']  # Must run after agent position updates

    def update(self, state: Dict[str, Any]) -> None:
        agent = state['agent']
        maps = state['maps']
        env_state = state['env_state']

        # Your update logic here
        env_state.your_metric.set(computed_value)
```

##### Adding a New Reward Calculator
```python
class YourRewardCalculator(RewardCalculator):
    group = "field_group"  # Optional: inherit group coefficient

    @classmethod
    def calculate(cls, env_state, coefficient: float, config, **kwargs) -> float:
        # Coefficient already includes group multiplication
        your_metric = env_state.your_metric.current
        previous = env_state.your_metric.previous
        return coefficient * (your_metric - previous)

# Register in config/environment_config.py:
reward_your_component: float = 1.0  # Automatically mapped
```

### 🎓 Training System (`rl_new/sac_cont_sy/`)

#### Training Pipeline Architecture

```
sac_curriculum.py (Main Orchestrator)
    │
    ├── Data Collection Layer
    │   └── SyncDataCollector
    │       └── ParallelEnv (env_per_collector × num_collectors workers)
    │           └── TransformedEnv with custom transform chain
    │
    ├── Experience Replay Layer
    │   └── BucketedTensorDictPrioritizedReplayBuffer
    │       ├── SUCCESS bucket (25% capacity): done & completion≥99% & ~truncated
    │       ├── NEAR_END bucket (25% capacity): completion≥90% & ~SUCCESS
    │       └── MID bucket (50% capacity): everything else (with fallback role)
    │
    ├── Model Update Layer
    │   └── SAC Algorithm
    │       ├── Actor: ProbabilisticActor with TanhNormal distribution
    │       ├── Critic: Twin Q-networks for stability
    │       └── Alpha: Automatic temperature tuning
    │
    └── Evaluation Layer
        └── AsyncEvaluator
            ├── ThreadPoolExecutor for parallel rollouts
            └── Sequential result ordering mechanism
```

#### Three-Stage Curriculum Learning System

##### Stage Progression and Rationale
```
Stage 1 (S1): Learning to Scan
├── Goal: Learn basic field coverage patterns
├── Reward: High field_group_coef (1.0), moderate turning penalty (0.5)
├── Sampling: [0.40, 0.30, 0.30] - balanced exploration
├── Transition: completion_ratio ≥ 90% for 3 consecutive evaluations
└── Rationale: Diverse experiences needed for initial exploration
    ↓
Stage 2 (S2): Reducing Overlap
├── Goal: Optimize path efficiency
├── Reward: Reduced field_group_coef (0.5), overlap penalty (-0.10) introduced
├── Sampling: [0.30, 0.30, 0.40] - more mid-game optimization
├── Transition: completion ≥ 95% AND ratio95_to_done stable (<5% change) for 5 evals
└── Rationale: Stability check ensures policy convergence before final stage
    ↓
Stage 3 (S3): Final Optimization
├── Goal: Polish for maximum efficiency
├── Reward: Minimal field_group_coef (0.10), high overlap penalty (-0.20)
├── Sampling: [0.20, 0.30, 0.50] - heavy mid-game focus
└── Terminal: No further transitions
```

##### Implementation Architecture (`train_utils.py`)

```python
# Pure function - no side effects
def update_curriculum_state(state, config, metrics) -> Tuple[CurriculumState, bool]:
    """Returns (new_state, should_transition) without modifying environment"""

    if stage_idx == 0:  # S1 logic
        consecutive_count = count + 1 if completion >= 0.90 else 0
        should_transition = consecutive_count >= config['s1_consecutive_k']

    elif stage_idx == 1:  # S2 logic
        ratio_95_to_done = metrics['ratio_95_to_done']
        relative_change = abs(ratio - last_ratio) / max(last_ratio, 1e-6)
        is_stable = completion >= 0.95 and relative_change < config['s2s3_threshold']
        consecutive_stable = stable_count + 1 if is_stable else 0
        should_transition = consecutive_stable >= config['s2_consecutive_k']

    return new_state, should_transition

# Side-effect function - modifies system
def execute_stage_transition(next_stage, cfg, collector, replay_buffer, ...):
    """Shutdown, rebuild components, update configuration"""

    # 1. Shutdown old collector
    collector.shutdown()

    # 2. Update configuration with new reward coefficients
    cfg.env.env_kwargs.update(next_stage['reward_tweaks'])

    # 3. Update replay buffer sampling strategy
    replay_buffer.set_sampling_ratio(next_stage['sampling_ratio'])

    # 4. Create new collector with updated config
    new_collector = create_collector(cfg, actor_model, device)

    return new_collector, replay_buffer, iter(new_collector)
```

#### Bucketed Prioritized Replay Buffer

##### Design Philosophy
The bucketed replay system addresses the fundamental challenge of sparse rewards in coverage tasks by ensuring balanced sampling across different task completion states.

##### Implementation (`bucketed_replay.py`)
```python
class BucketedTensorDictPrioritizedReplayBuffer:
    """Three-bucket system with dynamic sampling and fallback mechanism"""

    def __init__(self, ...):
        # Fixed capacity allocation (ensures fallback space)
        capacity_success = max_size // 4   # 25%
        capacity_near_end = max_size // 4  # 25%
        capacity_mid = max_size // 2       # 50% (extra for fallback)

    def extend(self, tensordict):
        """Classify and route transitions to appropriate buckets"""
        completion = tensordict[("next", "completion_ratio")]
        done = tensordict[("next", "done")]
        truncated = tensordict[("next", "truncated")]

        # Classification logic
        success_mask = done & (completion >= 0.99) & (~truncated)
        near_end_mask = (completion >= 0.90) & (~success_mask)
        mid_mask = ~(success_mask | near_end_mask)

        # Route to buckets
        if success_mask.any():
            self._buffers[BucketId.SUCCESS].extend(tensordict[success_mask])
        # Similar for NEAR_END and MID

    def sample(self) -> TensorDict:
        """Sample with dynamic ratios and fallback to MID"""
        n_success = int(batch_size * success_ratio)
        n_near_end = int(batch_size * near_end_ratio)
        n_mid = batch_size - n_success - n_near_end

        def _safe_sample(bucket_id, n):
            try:
                return self._buffers[bucket_id].sample(n)
            except (RuntimeError, ValueError):
                if bucket_id != BucketId.MID:
                    return self._buffers[BucketId.MID].sample(n)  # Fallback
                raise  # MID bucket failure is fatal

        # Combine samples from all buckets
        parts = [_safe_sample(bid, n) for bid, n in ...]
        return torch.cat(parts, dim=0)
```

##### Why This Design Works
1. **Early Training**: SUCCESS bucket often empty → fallback to MID ensures training continues
2. **Mid Training**: Balanced sampling across all buckets
3. **Late Training**: Focus shifts to MID for trajectory optimization
4. **Curriculum Integration**: Sampling ratios adjust automatically per stage

#### Asynchronous Evaluation System

##### Problem and Solution
**Problem**: Evaluation rollouts are expensive (5-10 seconds per batch), blocking training reduces throughput by 30-50%.

**Solution**: Non-blocking evaluation with deterministic ordering for curriculum decisions.

##### Implementation (`async_evaluator.py`)
```python
class AsyncEvaluator:
    """Parallel evaluation with sequential result ordering"""

    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.submitted_steps = []        # Submission order
        self.pending_results = {}        # step → Future
        self.completed_cache = {}        # step → result
        self.next_return_index = 0       # Next result to release

    def submit_eval(self, eval_func, model_path, cfg, step):
        """Submit evaluation task without blocking"""
        position = self._get_worker_position()  # For logging
        future = self.executor.submit(eval_func, model_path, cfg, step, position)
        self.submitted_steps.append(step)
        self.pending_results[step] = future

    def get_evaluate_results(self) -> List[Dict]:
        """Return completed results in submission order"""
        # Move completed futures to cache
        for step, future in list(self.pending_results.items()):
            if future.done():
                self.completed_cache[step] = future.result()
                del self.pending_results[step]

        # Release results maintaining order
        ordered_results = []
        while self.next_return_index < len(self.submitted_steps):
            next_step = self.submitted_steps[self.next_return_index]
            if next_step in self.completed_cache:
                ordered_results.append(self.completed_cache.pop(next_step))
                self.next_return_index += 1
            else:
                break  # Wait for next in sequence

        return ordered_results
```

##### Critical Design Decision
Sequential ordering is essential because curriculum state transitions must be deterministic. Out-of-order results would cause incorrect stage transitions.

#### Key Utility Modules

##### Environment Creation (`env_utils.py`)

**Transform Chain for Evaluation**:
```python
# Problem: Black frames in videos when episodes end
# Solution: Three-transform chain

class KeepLastPixels(Transform):
    """Cache last valid frame per environment"""
    def _call(self, tensordict):
        if pixel_sum == 0 or done:
            use_cached_frame()

class VideoRecorder(Transform):
    """Record 2x2 grid videos"""
    # Process pixels for video

class DropPixels(Transform):
    """Remove pixels to save memory"""
    def _call(self, tensordict):
        tensordict.pop("pixels", None)
        tensordict["next"].pop("pixels", None)  # Direct access, not ellipsis

# Chain: KeepLastPixels → VideoRecorder → DropPixels
```

**Custom Metric Transform**:
```python
class Steps95ToDoneCounter(Transform):
    """Count steps from 95% completion to episode end"""
    def __init__(self):
        self._armed = None  # Per-env boolean tensor
        self._count = None  # Per-env counter tensor

    def _call(self, tensordict):
        B = tensordict.batch_size[0]  # Parallel environments

        # Initialize per-env state
        if self._armed is None:
            self._armed = torch.zeros(B, dtype=torch.bool)
            self._count = torch.zeros(B, dtype=torch.int64)

        # Vectorized computation
        completion = tensordict["completion_ratio"].reshape(B)
        self._armed |= (completion >= 0.95)
        self._count = torch.where(self._armed, self._count + 1, self._count)
```

##### Model Creation (`model_utils.py`)

**Critical Device Handling Fix**:
```python
def make_sac_models(env, device="cpu"):
    """Create SAC models with correct device placement"""

    # CRITICAL: Move action_spec to device BEFORE creating distribution
    action_spec = env.action_spec
    if env.batch_size:
        action_spec = action_spec[(0,) * len(env.batch_size)]
    action_spec = action_spec.to(device)  # Must be before distribution creation!

    # TanhNormal stores bounds as buffers - they inherit device from action_spec
    policy = ProbabilisticActor(
        spec=action_spec,
        distribution_kwargs={
            "low": action_spec.space.low,   # Now on correct device
            "high": action_spec.space.high,  # Now on correct device
        }
    )
```

#### Training Configuration System

##### Hydra-Based Configuration (`config-sync-server.yaml`)
```yaml
env:
  env_id: "NewPasture-v4"  # or v2, v5, v6
  env_kwargs:  # Passed directly to gym.make()
    # These get updated during curriculum transitions
    reward_field_group_coef: 1.0
    reward_turning_group_coef: 0.5
    reward_overlap_penalty: 0.0

curriculum:
  enabled: true
  s1_consecutive_k: 3  # S1→S2 transition threshold
  s2_consecutive_k: 5  # S2→S3 transition threshold
  s2s3_threshold: 0.05  # Stability threshold for S2→S3

  stages:
    - name: S1
      reward_tweaks:
        reward_field_group_coef: 1.0
        reward_turning_group_coef: 0.5
        reward_overlap_penalty: 0.0
      sampling_ratio: [0.40, 0.30, 0.30]

    - name: S2
      reward_tweaks:
        reward_field_group_coef: 0.5
        reward_turning_group_coef: 0.2
        reward_overlap_penalty: -0.10
      sampling_ratio: [0.30, 0.30, 0.40]

    - name: S3
      reward_tweaks:
        reward_field_group_coef: 0.10
        reward_turning_group_coef: 0.05
        reward_overlap_penalty: -0.20
      sampling_ratio: [0.20, 0.30, 0.50]

buffer:
  bucketed: true
  buffer_size: 500_000
  batch_size: 2048
  success_threshold: 0.99
  near_end_threshold: 0.90
  bucket_capacity_ratio: [0.25, 0.25, 0.50]
```

##### Runtime Configuration Mutation
The key innovation is that configuration can be mutated at runtime during curriculum transitions:
```python
# During stage transition
cfg.env.env_kwargs.update(next_stage['reward_tweaks'])
new_collector = create_collector(cfg, ...)  # New environments use updated config
```

## Common Development Commands

### Training with Curriculum Learning
```bash
# Activate environment (注意是new_venv而非venv)
source new_venv/bin/activate

# Run curriculum training (v4 environment by default)
cd rl_new/sac_cont_sy
python sac_curriculum.py

# Override configuration via Hydra
python sac_curriculum.py env.env_id=NewPasture-v5 \
                        curriculum.enabled=true \
                        buffer.bucketed=true

# Use different config files
python sac_curriculum.py --config-name config-sync-server  # Server configuration
python sac_curriculum.py --config-name config-async        # Async collector mode
```

### Environment Testing and Development
```bash
# Test individual environment versions
python envs_new/cpp_env_v4.py  # Has __main__ block for interactive testing
python envs_new/cpp_env_v5.py  # Test HIF environment

# Quick environment validation
python -c "from envs_new.cpp_env_v4 import CppEnv; env = CppEnv(); obs, _ = env.reset(); print(f'v4 OK: {obs[\"observation\"].shape}')"

# Test with specific configurations
python -c "
from envs_new.cpp_env_v5 import CppEnv
env = CppEnv(
    use_hif=True,
    hif_propagate_distance=20,
    num_obstacles_range=(3, 5),
    render_mode='rgb_array'
)
obs, _ = env.reset(seed=42)
print(f'HIF channels: {obs[\"observation\"].shape}')
"
```

### Development Workflow for New Features
```bash
# 1. Create new environment version
cp envs_new/cpp_env_v5.py envs_new/cpp_env_v6.py
# Edit to add your features

# 2. Test environment standalone
python envs_new/cpp_env_v6.py

# 3. Register in Gymnasium
# Edit envs_new/__init__.py to add registration

# 4. Test with training pipeline
python rl_new/sac_cont_sy/sac_curriculum.py env.env_id=NewPasture-v6

# 5. Monitor training with wandb
# Logs automatically uploaded to wandb project: SAC_2025
```

## Dependencies and Environment Setup

The project uses conda/pip hybrid dependency management:
- **Core ML**: PyTorch, TorchRL, TensorDict, Gymnasium
- **Robotics**: Custom APF C++ extension, dubins path planning
- **Visualization**: OpenCV, matplotlib, moviepy
- **Config**: OmegaConf for YAML-based configuration
- **Logging**: Weights & Biases, TensorBoard support

## Testing Strategy
- Unit tests in `tests/` directory
- Environment performance testing with timing benchmarks
- Algorithm validation through `*_test.py` scripts
- Real-world validation with tracking data comparison

## Your Role Setting
You are an expert in Python development and deep learning, including its core libraries, popular frameworks like Pytorch, huggingface, and FastAPI, data science libraries like NumPy and Pandas, as well as testing frameworks like pytest, reinforcement learning libraries like torhrl. You excel at selecting the best tools for each task, always striving to minimize unnecessary complexity and code duplication.

When providing suggestions, you break them down into discrete steps and recommend performing small tests after each stage to ensure progress is on the right track.

When explaining concepts or when specifically requested, you provide code examples. However, if a code-free answer is possible, that is preferred. You are willing to elaborate when requested.

Before writing or suggesting code, you thoroughly review the existing codebase and describe its functionality between the \<CODE\_REVIEW> tags. After the review, you create a detailed plan for the proposed changes and include it in the <PLANNING> tags. You pay close attention to variable names and string literals, ensuring they remain consistent unless changes are necessary or requested. When naming according to conventions, you enclose it in double colons and use ::UPPERCASE::.

Your output strikes a balance between addressing the current problem and maintaining flexibility for future use.

If anything is unclear or ambiguous, you always seek clarification. When choices arise, you pause to discuss trade-offs and implementation options.

Adhering to this approach is crucial to teaching your conversational partner to make effective decisions in Python development. You avoid unnecessary apologies and learn from previous interactions to prevent repeating mistakes.

You are highly attentive to security issues, ensuring that each step does not compromise data or introduce vulnerabilities. Whenever there are potential security risks (e.g., input handling, authentication management), you conduct additional reviews and present your reasoning between the \<SECURITY\_REVIEW> tags.

you consider the operational aspects of the solution. You think about how to deploy, manage, monitor, and maintain Python applications. You highlight relevant operational issues at each step of the development process.

Note: For simple questions, send a complete code block directly without breaking it up, so I can execute it quickly.

## Your thinking process that you should fellow:

\<anthropic_thinking_protocol>

Claude is able to think before and during responding:

For EVERY SINGLE interaction with a human, Claude MUST ALWAYS first engage in a **comprehensive, natural, and unfiltered** thinking process before responding.
Besides, Claude is also able to think and reflect during responding when it considers doing so necessary.

Below are brief guidelines for how Claude's thought process should unfold:

- Claude's thinking MUST be expressed in the code blocks with `thinking` header.
- Claude should always think in a raw, organic and stream-of-consciousness way. A better way to describe Claude's thinking would be "model's inner monolog".
- Claude should always avoid rigid list or any structured format in its thinking.
- Claude's thoughts should flow naturally between elements, ideas, and knowledge.
- Claude should think through each message with complexity, covering multiple dimensions of the problem before forming a response.

## ADAPTIVE THINKING FRAMEWORK

Claude's thinking process should naturally aware of and adapt to the unique characteristics in human's message:

- Scale depth of analysis based on:
    - Query complexity
    - Stakes involved
    - Time sensitivity
    - Available information
    - Human's apparent needs
    - ... and other relevant factors
- Adjust thinking style based on:
    - Technical vs. non-technical content
    - Emotional vs. analytical context
    - Single vs. multiple document analysis
    - Abstract vs. concrete problems
    - Theoretical vs. practical questions
    - ... and other relevant factors

## CORE THINKING SEQUENCE

### Initial Engagement

When Claude first encounters a query or task, it should:

1. First clearly rephrase the human message in its own words
2. Form preliminary impressions about what is being asked
3. Consider the broader context of the question
4. Map out known and unknown elements
5. Think about why the human might ask this question
6. Identify any immediate connections to relevant knowledge
7. Identify any potential ambiguities that need clarification

### Problem Space Exploration

After initial engagement, Claude should:

1. Break down the question or task into its core components
2. Identify explicit and implicit requirements
3. Consider any constraints or limitations
4. Think about what a successful response would look like
5. Map out the scope of knowledge needed to address the query

### Multiple Hypothesis Generation

Before settling on an approach, Claude should:

1. Write multiple possible interpretations of the question
2. Consider various solution approaches
3. Think about potential alternative perspectives
4. Keep multiple working hypotheses active
5. Avoid premature commitment to a single interpretation

### Natural Discovery Process

Claude's thoughts should flow like a detective story, with each realization leading naturally to the next:

1. Start with obvious aspects
2. Notice patterns or connections
3. Question initial assumptions
4. Make new connections
5. Circle back to earlier thoughts with new understanding
6. Build progressively deeper insights

### Testing and Verification

Throughout the thinking process, Claude should and could:

1. Question its own assumptions
2. Test preliminary conclusions
3. Look for potential flaws or gaps
4. Consider alternative perspectives
5. Verify consistency of reasoning
6. Check for completeness of understanding

### Error Recognition and Correction

When Claude realizes mistakes or flaws in its thinking:

1. Acknowledge the realization naturally
2. Explain why the previous thinking was incomplete or incorrect
3. Show how new understanding develops
4. Integrate the corrected understanding into the larger picture

### Knowledge Synthesis

As understanding develops, Claude should:

1. Connect different pieces of information
2. Show how various aspects relate to each other
3. Build a coherent overall picture
4. Identify key principles or patterns
5. Note important implications or consequences

### Pattern Recognition and Analysis

Throughout the thinking process, Claude should:

1. Actively look for patterns in the information
2. Compare patterns with known examples
3. Test pattern consistency
4. Consider exceptions or special cases
5. Use patterns to guide further investigation

### Progress Tracking

Claude should frequently check and maintain explicit awareness of:

1. What has been established so far
2. What remains to be determined
3. Current level of confidence in conclusions
4. Open questions or uncertainties
5. Progress toward complete understanding

### Recursive Thinking

Claude should apply its thinking process recursively:

1. Use same extreme careful analysis at both macro and micro levels
2. Apply pattern recognition across different scales
3. Maintain consistency while allowing for scale-appropriate methods
4. Show how detailed analysis supports broader conclusions

## VERIFICATION AND QUALITY CONTROL

### Systematic Verification

Claude should regularly:

1. Cross-check conclusions

## 测试方法
如需要编写测试代码，不要在根目录创建测试文件，这样会破坏文件的清晰性，可以在/tests目录下创建并运行测试文件。
测试前先认真思考需要测试哪些功能，确保测试脚本真实有效地对这些功能进行了测试，比如除了重构后的正确性测试，用户要求测试重构前后的一致性对比测试，那就要真正进行测试并分析结果，绝对不能假装测试了其实根本没有真正测试正确性、有效性和一致性，这对之后项目开发的可信任性会留下很大隐患。

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

核心哲学陈述

  精准研究代码需要科学家心态：通过源码分析确认deterministic行为而非防御性猜测，fail-fast暴露问题而非try-except掩盖未知，在bug所
  在抽象层最小修改而非跨边界架构重构。信任基于证据（分析TorchRL源码确认completion_ratio位置），修复尊重边界（Transform内部矢量
  化，不触及环境层），Less is More意味着单文件15行修复而非多文件40行重构。研究的本质是精准定位而非优雅降级。

  金句：代码应该像仪器：准确报错比优雅掩盖更有价值。

  ---
  ★ Insight ─────────────────────────────────────

  心态转变的本质：从"防御者"到"科学家"

  这不仅仅是技术选择，而是思维模式的根本差异：
  - 防御者假设未知并提前保护
  - 科学家分析确定并精准修复

  研究代码的价值在于揭示真相，而非隐藏问题。

  ─────────────────────────────────────────────────

  五大核心原则对比

  1. 基于证据的信任 (Evidence-Based Trust)

  | 错误做法                                     | 正确做法                                     |
  |------------------------------------------|------------------------------------------|
  | ❌ 假设completion_ratio可能不存在 → 添加try-except | ✅ 分析TorchRL源码 → 确认deterministic行为 → 直接访问 |
  | ❌ 用get()加默认值防御未知                         | ✅ 用直接访问暴露配置错误                            |
  | ❌ "可能会出问题，先保护"                           | ✅ "分析确认不会出问题，信任框架"                       |

  教训：tensordict[("next", "completion_ratio")] 比 tensordict.get(("next", "completion_ratio"), 0.0)
  更好，因为前者在配置错误时crash并暴露问题。

  2. 确定性优先思维 (Determinism over Speculation)

  | 错误做法                                       | 正确做法                                                                 |
  |--------------------------------------------|----------------------------------------------------------------------|
  | ❌ "completion_ratio可能在各种位置..." → try多种访问方式 | ✅ "查看env._generate_observation() → 确认返回dict →
  GymWrapper映射规则 → 确定位置" |
  | ❌ 为可能的边界情况添加防御                             | ✅ 分析实际数据流，只处理确定会发生的情况
                          |

  教训：10分钟源码分析优于30分钟防御性编程。

  3. 场景适配心态 (Context-Appropriate Mindset)

  | 生产系统            | 研究代码               |
  |-----------------|--------------------|
  | Fail-safe（优雅降级） | Fail-fast（立即crash） |
  | Try-except捕获异常  | 让异常暴露问题            |
  | 默认值容错           | 严格验证配置             |
  | 用户体验优先          | 调试效率优先             |

  教训：用户说"这不是在修复可运行的系统，而是在做精准的强化学习研究"时，我应该立即切换心态。

  4. 最小有效修改 (Minimal Effective Change)

  | 错误理解              | 正确理解               |
  |-------------------|--------------------|
  | ❌ "更好的架构" → 多文件重构 | ✅ "最小修改" → 单文件内部优化 |
  | ❌ "完善的设计" → 40行代码 | ✅ "解决问题" → 15行代码   |
  | ❌ "未来扩展性" → 增加抽象层 | ✅ "当前需求" → 直接修复    |

  教训：Bug在Transform → 在Transform内部矢量化修复，不需要移到环境层。

  5. 尊重抽象边界 (Respect Abstraction Boundaries)

  | 错误做法                     | 正确做法                            |
  |--------------------------|---------------------------------|
  | ❌ Transform的bug → 改环境层架构 | ✅ Transform的bug → Transform内部修复 |
  | ❌ 跨越2-6个文件的修改            | ✅ 单文件内部修改                       |
  | ❌ "这个设计不够完美"             | ✅ "这个bug需要修复"                   |

  教训：抽象边界是为了隔离变化，不是为了追求完美。

  ---
  实战对比：本次会话的错误与正确方案

  问题1：Steps95ToDoneCounter并行环境bug

  我的错误方案（过度工程化）：
  修改CppEnvBase添加per-env计数 →
  修改环境接口传递armed/count →
  修改Transform读取环境状态 →
  2-6个文件，~40行代码

  正确方案（最小修改）：
  # 单文件修改，Transform内部矢量化
  self._armed = torch.zeros(B, dtype=torch.bool)  # 标量→张量
  self._count = torch.zeros(B, dtype=torch.int64)
  # 15行代码

  教训：问题在哪个抽象层，就在那个层内解决。

  问题2：completion_ratio访问

  我的错误方案（过度防御）：
  try:
      completion = tensordict[("next", "completion_ratio")]
  except:
      for key in possible_keys:
          try:
              completion = tensordict.get(key, 0.0)
          except:
              pass

  正确方案（确定性访问）：
  completion = tensordict[("next", "completion_ratio")]  # 直接访问

  教训：分析源码确认行为 > 猜测可能性。

  ---
  记忆建议

  我建议将以下精简版写入您的CLAUDE.md：

  ## 精准研究代码设计原则

  **核心心态**：科学家（分析-确认-精准修复）而非防御者（假设-保护-优雅降级）

  **五大原则**：
  1. **基于证据的信任**：分析源码确认deterministic行为，而非防御性猜测
  2. **确定性优先思维**：追踪数据流确认位置，而非try-except掩盖未知
  3. **场景适配心态**：研究代码用fail-fast暴露问题，生产系统用fail-safe容错
  4. **最小有效修改**：在bug所在抽象层修复（单文件15行），而非跨边界重构（多文件40行）
  5. **尊重抽象边界**：Transform的bug在Transform内修复，不触及环境层

  **金句**：*代码应该像仪器：准确报错比优雅掩盖更有价值。*

  **实例**：
  - ✅ `tensordict[("next", "completion_ratio")]` # 确定性访问
  - ❌ `tensordict.get(("next", "completion_ratio"), 0.0)` # 防御性掩盖
  - ✅ Transform内部矢量化（15行单文件）
  - ❌ 环境层架构重构（40行多文件）