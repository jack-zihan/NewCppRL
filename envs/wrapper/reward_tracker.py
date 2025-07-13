import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from pathlib import Path
import pandas as pd


class RewardTracker(gym.Wrapper):
    """A wrapper that tracks detailed reward components by capturing state changes."""

    def __init__(self, env):
        super().__init__(env)
        self.reward_history = []
        self.current_episode_rewards = []
        self._step_count = 0

    def _capture_state(self, env) -> Dict[str, Any]:
        """Capture the current state of the environment."""
        return {
            'weed_num': env.map_weed.sum(dtype=np.int32),
            'frontier_area': env.map_frontier.sum(dtype=np.int32),
            'frontier_tv': self._calculate_frontier_tv(env),
            'steer': env.steer_t,
            'weed_num_t': env.weed_num_t,
            'frontier_area_t': env.frontier_area_t,
            'frontier_tv_t': env.frontier_tv_t,
            'steer_t': env.steer_t,
        }

    def _calculate_frontier_tv(self, env) -> int:
        """Calculate total variation of frontier map."""
        try:
            from envs.utils import total_variation
            return total_variation(env.map_frontier.astype(np.int32))
        except ImportError:
            # Fallback implementation if total_variation is not available
            frontier = env.map_frontier.astype(np.int32)
            tv = 0
            # Horizontal differences
            tv += np.abs(frontier[1:, :] - frontier[:-1, :]).sum()
            # Vertical differences
            tv += np.abs(frontier[:, 1:] - frontier[:, :-1]).sum()
            return tv

    def step(self, action):
        """Override step to capture reward details."""
        # Get the actual environment
        actual_env = self.unwrapped

        # Capture state before step
        state_before = self._capture_state(actual_env)

        # Store the original get_reward method
        original_get_reward = actual_env.get_reward

        # Variables to capture reward components
        reward_components = {}

        # Create an intercepting get_reward
        def intercepting_get_reward(steer_tp1, x_t, y_t, x_tp1, y_tp1):
            # Calculate state values that will be used in reward calculation
            weed_num_tp1 = actual_env.map_weed.sum(dtype=np.int32)
            frontier_area_tp1 = actual_env.map_frontier.sum(dtype=np.int32)
            frontier_tv_tp1 = self._calculate_frontier_tv(actual_env)

            # Calculate reward components using the same logic as original
            reward_const = -0.1

            # Turning rewards
            reward_turn_gap = -0.5 * abs(steer_tp1 - actual_env.steer_t) / actual_env.w_range.max
            reward_turn_direction = -0.30 * (0. if (steer_tp1 * actual_env.steer_t >= 0
                                                    or (steer_tp1 == 0 and actual_env.steer_t == 0))
                                             else 1.)
            reward_turn_self = 0.25 * (0.4 - abs(steer_tp1 / actual_env.w_range.max) ** 0.5)
            reward_turn = 1 * (reward_turn_gap + reward_turn_direction + reward_turn_self)

            # Frontier rewards
            reward_frontier_coverage = (actual_env.frontier_area_t - frontier_area_tp1) / (
                    2 * actual_env.agent.width * actual_env.v_range.max)
            reward_frontier_tv = 0.5 * (actual_env.frontier_tv_t - frontier_tv_tp1) / actual_env.v_range.max
            reward_frontier = 0.125 * (reward_frontier_coverage + reward_frontier_tv)

            # Weed reward
            reward_weed = 20.0 * (actual_env.weed_num_t - weed_num_tp1)

            # Call original method to get extra reward and total
            total_reward = original_get_reward(steer_tp1, x_t, y_t, x_tp1, y_tp1)

            # Calculate extra reward as the difference
            reward_extra = total_reward - (reward_const + reward_turn + reward_frontier + reward_weed)

            # Store all components
            reward_components.update({
                'total': float(total_reward),
                'const': float(reward_const),
                'turn': float(reward_turn),
                'turn_gap': float(reward_turn_gap),
                'turn_direction': float(reward_turn_direction),
                'turn_self': float(reward_turn_self),
                'frontier': float(reward_frontier),
                'frontier_coverage': float(reward_frontier_coverage),
                'frontier_tv': float(reward_frontier_tv),
                'weed': float(reward_weed),
                'extra': float(reward_extra),
                'timestep': actual_env.t,
                'steer_t': actual_env.steer_t,
                'steer_tp1': steer_tp1,
                'weed_num_t': actual_env.weed_num_t,  
                'weed_num_tp1': weed_num_tp1,
                'frontier_area_t': actual_env.frontier_area_t,
                'frontier_area_tp1': frontier_area_tp1,
                'frontier_tv_t': actual_env.frontier_tv_t,
                'frontier_tv_tp1': frontier_tv_tp1,
            })

            return total_reward

        # Temporarily replace the method
        actual_env.get_reward = intercepting_get_reward

        try:
            # Call the original step method
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Add reward details to info
            if reward_components:
                info['reward_details'] = reward_components
                self.current_episode_rewards.append(reward_components)

            self._step_count += 1

        finally:
            # Always restore the original method
            actual_env.get_reward = original_get_reward

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and save episode rewards."""
        # Save the current episode rewards if any
        if self.current_episode_rewards:
            self.reward_history.append(self.current_episode_rewards.copy())

        # Clear current episode rewards
        self.current_episode_rewards = []
        self._step_count = 0

        # Reset the environment
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def save_rewards(self, filepath: Union[str, Path]):
        """Save all reward history to a CSV file."""
        # Convert Path to string if necessary
        filepath = str(filepath)
        # Include current episode if it has data
        if self.current_episode_rewards:
            all_history = self.reward_history + [self.current_episode_rewards]
        else:
            all_history = self.reward_history

        if not all_history:
            print("No reward history to save.")
            return

        # Flatten all episodes into a single DataFrame
        all_rewards = []
        for episode_idx, episode_rewards in enumerate(all_history):
            for step_idx, reward_dict in enumerate(episode_rewards):
                reward_dict['episode'] = episode_idx
                reward_dict['step'] = step_idx
                all_rewards.append(reward_dict)

        df = pd.DataFrame(all_rewards)

        # Reorder columns for better readability
        first_cols = ['episode', 'step', 'timestep', 'total', 'const', 'turn', 'frontier', 'weed', 'extra']
        other_cols = [col for col in df.columns if col not in first_cols]
        df = df[first_cols + other_cols]

        df.to_csv(filepath, index=False)
        print(f"Saved {len(all_rewards)} reward records from {len(all_history)} episodes to {filepath}")

    def get_episode_summary(self, episode_idx: int = -1) -> Dict[str, float]:
        """Get summary statistics for a specific episode."""
        # Include current episode if requested
        if episode_idx == -1 and self.current_episode_rewards:
            episode_rewards = self.current_episode_rewards
        elif episode_idx < len(self.reward_history):
            episode_rewards = self.reward_history[episode_idx]
        else:
            return {}

        if not episode_rewards:
            return {}

        # Calculate summary statistics
        summary = {}
        reward_keys = ['total', 'const', 'turn', 'frontier', 'weed', 'extra']

        for key in reward_keys:
            values = [r[key] for r in episode_rewards]
            summary[f'{key}_sum'] = sum(values)
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)

        # Add episode length
        summary['episode_length'] = len(episode_rewards)

        return summary

    def plot_rewards(self, episode_idx: int = -1, save_path: Union[str, Path] = None):
        """Plot reward components over time for a specific episode."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Cannot plot rewards.")
            return

        # Get episode data
        if episode_idx == -1 and self.current_episode_rewards:
            episode_rewards = self.current_episode_rewards
        elif episode_idx < len(self.reward_history):
            episode_rewards = self.reward_history[episode_idx]
        else:
            print(f"Episode {episode_idx} not found.")
            return

        if not episode_rewards:
            print("No rewards to plot.")
            return

        # Extract data
        steps = [r['step'] for r in episode_rewards]
        reward_types = ['total', 'const', 'turn', 'frontier', 'weed', 'extra']

        # Create subplots
        fig, axes = plt.subplots(len(reward_types), 1, figsize=(12, 8), sharex=True)

        for idx, reward_type in enumerate(reward_types):
            values = [r[reward_type] for r in episode_rewards]
            axes[idx].plot(steps, values, label=reward_type.capitalize())
            axes[idx].set_ylabel(reward_type.capitalize())
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()

        axes[-1].set_xlabel('Step')
        plt.suptitle(f'Reward Components - Episode {episode_idx}')
        plt.tight_layout()

        if save_path:
            # Convert Path to string if necessary
            plt.savefig(str(save_path))
            print(f"Saved reward records trajectory episodes to {str(save_path)}")
        else:
            plt.show()