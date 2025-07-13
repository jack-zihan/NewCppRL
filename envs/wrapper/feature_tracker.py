import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple
import cv2
from envs.utils import get_map_pasture_larger, total_variation_mat

class FeatureTracker(gym.Wrapper):
    """A wrapper that captures environment features for visualization and analysis."""

    def __init__(self, env):
        super().__init__(env)
        self.current_features = {}
        self._step_count = 0
        self._episode_count = 0

    def _capture_features(self, env) -> Dict[str, np.ndarray]:
        """Capture all relevant map features from the environment."""
        features = {}

        # Get the actual environment
        actual_env = env.unwrapped

        # Capture rendered map
        features['rendered_map'] = actual_env.render_map()

        # Get maps before APF processing
        if hasattr(actual_env, 'get_maps_and_mask'):
            # Store original use_apf state
            original_use_apf = actual_env.use_apf

            # Get maps without APF
            actual_env.use_apf = False
            maps_no_apf, _ = actual_env.get_maps_and_mask()

            # Extract individual maps before APF
            if actual_env.noise_weed and actual_env.np_random.uniform() < actual_env.noise_weed:
                map_weed_ = actual_env.map_weed_noisy
            else:
                map_weed_ = actual_env.map_weed

            # Import total_variation_mat
            try:
                from envs.utils import total_variation_mat
            except ImportError:
                # Fallback implementation
                def total_variation_mat(images: np.ndarray) -> np.ndarray:
                    mask_tv_cols = images[1:, :] - images[:-1, :] != 0
                    mask_tv_cols = np.pad(mask_tv_cols, pad_width=[[0, 1], [0, 0]], mode='constant')
                    mask_tv_rows = images[:, 1:] - images[:, :-1] != 0
                    mask_tv_rows = np.pad(mask_tv_rows, pad_width=[[0, 0], [0, 1]], mode='constant')
                    mask_tv = np.logical_or(mask_tv_rows, mask_tv_cols)
                    return mask_tv

            # Calculate raw maps (before APF)
            features['frontier_full'] = np.logical_and(
                total_variation_mat(actual_env.map_frontier_full),
                actual_env.map_mist
            ).astype(np.float32)
            features['weed_full'] = np.logical_and(
                total_variation_mat(actual_env.map_weed_ori),
                actual_env.map_mist
            ).astype(np.float32)
            features['frontier_raw'] = np.logical_and(
                total_variation_mat(actual_env.map_frontier),
                actual_env.map_mist
            ).astype(np.float32)
            features['obstacle_raw'] = np.logical_and(
                total_variation_mat(actual_env.map_obstacle),
                actual_env.map_mist
            ).astype(np.float32)
            features['weed_raw'] = np.logical_and(
                map_weed_,
                np.logical_not(actual_env.map_frontier)
            ).astype(np.float32)
            features['trajectory_raw'] = actual_env.map_trajectory.astype(np.float32)

            # Restore original use_apf and get processed maps
            actual_env.use_apf = original_use_apf
            maps_with_apf, _ = actual_env.get_maps_and_mask()

            # Extract APF-processed maps
            features['frontier_apf'] = maps_with_apf[:, :, 0]
            features['mist'] = maps_with_apf[:, :, 1]
            features['obstacle_apf'] = maps_with_apf[:, :, 2]
            features['weed_apf'] = maps_with_apf[:, :, 3]
            if maps_with_apf.shape[-1] > 4:
                features['trajectory_apf'] = maps_with_apf[:, :, 4]

        # Add metadata
        features['metadata'] = {
            'step': self._step_count,
            'episode': self._episode_count,
            'agent_position': actual_env.agent.position,
            'agent_direction': actual_env.agent.direction,
            'weed_count': actual_env.map_weed.sum(),
            'frontier_area': actual_env.map_frontier.sum(),
        }

        return features

    def step(self, action):
        """Override step to capture features after each action."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Capture features after step
        self.current_features = self._capture_features(self.env)
        self._step_count += 1

        # Add feature availability flag to info
        info['features_available'] = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and update counters."""
        obs, info = self.env.reset(**kwargs)

        self._step_count = 0
        self._episode_count += 1

        # Capture initial features
        self.current_features = self._capture_features(self.env)

        return obs, info

    def get_current_features(self) -> Dict[str, Any]:
        """Get the current captured features."""
        return self.current_features.copy()

    def render(self):
        """Pass through render call to wrapped environment."""
        return self.env.render()