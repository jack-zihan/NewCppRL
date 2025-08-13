# Pasture-v4 and Pasture-v5 Environments

## Overview

Two new Gymnasium environments for robotic navigation and path planning without weed rewards:

- **Pasture-v4**: No mist, no weed rewards, includes frontier_hifs map, no multi-scale observation
- **Pasture-v5**: Same as v4 but with multi-scale SGCNN observation

## Key Features

### Common Features
- ✅ No weed rewards in the reward function  
- ✅ Termination based on frontier coverage (not weed removal)
- ✅ Includes frontier_hifs map layer from `envs/maps/hifs/` directory
- ✅ Coverage rate tracking instead of weed ratio
- ✅ No mist in the environment

### Pasture-v4 Specifics
- Observation shape: `(4, 128, 128)`
- Channels: frontier, frontier_hifs, obstacle, trajectory
- Direct pixel-level observation

### Pasture-v5 Specifics  
- Observation shape: `(20, 16, 16)` or `(16, 16, 16)`
- Multi-scale SGCNN features
- 4 scales of spatial pyramid pooling

## Installation

The environments are already registered in `envs/__init__.py`:

```python
import gymnasium as gym
import envs

# Create environments
env_v4 = gym.make("Pasture-v4")
env_v5 = gym.make("Pasture-v5")
```

## Usage Example

```python
import gymnasium as gym
import envs

# Using Pasture-v4 (no multi-scale)
env = gym.make("Pasture-v4")
obs, info = env.reset()

print(f"Observation shape: {obs['observation'].shape}")  # (4, 128, 128)
print(f"Initial coverage: {info['coverage_rate']:.2%}")

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    # Monitor progress
    coverage = info['coverage_rate']
    print(f"Coverage: {coverage:.2%}, Reward: {reward:.4f}")
    
    if done:
        if info.get('crashed'):
            print("Episode ended: Collision")
        elif info.get('finished'):
            print("Episode ended: Frontier fully covered!")

env.close()
```

## Map Structure

### Frontier Maps
- Primary frontier maps: `envs/maps/1-400/farmland_*.png`
- HIFS frontier maps: `envs/maps/hifs/farmland_*.png`

If a HIFS map is not found, the environment will use the primary frontier map as a fallback.

## Reward Components

The reward function excludes weed rewards and consists of:

1. **Constant penalty**: -0.1 per step
2. **Frontier coverage reward**: Based on frontier area reduction  
3. **Frontier total variation reward**: Rewards smooth coverage
4. **Turn penalty**: Penalizes excessive turning (currently scaled to 0)
5. **Extra rewards**: From parent class (e.g., APF rewards if enabled)
6. **Collision penalty**: -399 on collision
7. **Completion bonus**: +500 when frontier fully covered

## Info Dictionary

The `info` dictionary returned by `step()` and `reset()` contains:

- `coverage_rate`: Percentage of frontier covered (0.0 to 1.0)
- `crashed`: Boolean indicating collision
- `finished`: Boolean indicating frontier fully covered

## Testing

Run the comprehensive test suite:

```bash
python tests/test_cpp_env_v4_v5.py
```

Quick test:

```bash
python tests/quick_test_v4_v5.py
```

## Implementation Details

The implementation is in `envs/cpp_env_v4.py`:

- `CppEnvV4`: Base class for v4 environment
- `CppEnvV5`: Inherits from V4 but enables multi-scale observation

Key overridden methods:
- `generate_frontier_maps()`: Loads frontier_hifs maps
- `get_maps_and_mask()`: Returns 4-channel observation
- `get_reward()`: Completely rewritten to exclude weed rewards
- `step()`: Modified termination condition  
- `observation()`: Updates weed_ratio to coverage_rate
- `reset()`: Adds coverage_rate to info

## Notes

1. The environment still generates weeds internally but they don't affect rewards or termination
2. The `weed_ratio` key in observations is retained for compatibility but contains coverage_rate
3. Some warning messages about missing HIFS maps are expected - the environment handles this gracefully