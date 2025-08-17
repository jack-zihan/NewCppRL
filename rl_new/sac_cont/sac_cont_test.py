from pathlib import Path
import gymnasium as gym
import torch
import yaml
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from gymnasium.wrappers import HumanRendering
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
from datetime import datetime

import envs  # noqa
from envs.wrapper.reward_tracker import RewardTracker
from envs.wrapper.feature_tracker import FeatureTracker


class SACTestSession:
    """Integrated SAC testing session with reward tracking and feature visualization."""
    
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(__file__).parent.parent.parent
        self.episode_returns = []
        self.frame_counter = 0
        self.fig = None  # Current matplotlib figure
        
        # Initialize pygame
        pygame.init()
        
        # Setup matplotlib
        if self.config['log_feature']:
            matplotlib.use('TkAgg')
            plt.ion()
        
        self._setup_directories()
        self._load_model()
        self._setup_environment()

    def _setup_directories(self):
        """Setup logging directories."""
        if self.config['log_reward']:
            self.reward_dir = self.base_dir / 'logs' / 'reward_analysis'
            self.reward_dir.mkdir(parents=True, exist_ok=True)
            
        if self.config['log_feature']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.feature_dir = self.base_dir / 'logs' / 'feature_analysis' / timestamp
            self.feature_dir.mkdir(parents=True, exist_ok=True)
            print(f"Features will be saved to: {self.feature_dir}")
    
    def _load_model(self):
        """Load the SAC model."""
        self.device = self.config['device']
        actor_critic = torch.load(self.config['model_path']).to(self.device)
        self.actor = actor_critic[0].to(self.device)
        
    def _setup_environment(self):
        """Setup environment with appropriate wrappers."""
        # Load environment config
        cfg = DictConfig(yaml.load(open(f'{self.base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
        
        # Create base environment
        self.env = gym.make(
            render_mode='rgb_array' if self.config['render'] else None,
            **cfg.env.params,
        )
        
        # Apply wrappers in order (IMPORTANT: order matters for compatibility)
        if self.config['log_reward']:
            self.env = RewardTracker(self.env)
            
        if self.config['log_feature']:
            self.env = FeatureTracker(self.env)
            
        if self.config['render']:
            self.env = HumanRendering(self.env)
    
    def visualize_features(self, features, save_path=None):
        """Visualize all captured features in a comprehensive layout."""
        rendered_map = features['rendered_map']
        metadata = features['metadata']
        
        # Get configurable figure size (default to reasonable size)
        figsize = self.config.get('visualization_figsize', (12, 9))
        
        # Create figure with interactive capabilities
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Environment Features - Episode {metadata['episode']}, Step {metadata['step']}", fontsize=14)
        
        # Enable interactive toolbar and resizable window
        if hasattr(fig.canvas, 'toolbar_visible'):
            fig.canvas.toolbar_visible = True
        
        # Make window resizable and add zoom/pan tools
        manager = fig.canvas.manager
        if hasattr(manager, 'window'):
            try:
                manager.window.wm_title(f"Feature Visualization - Episode {metadata['episode']}")
                # Enable resizing
                manager.window.resizable(True, True)
            except:
                pass  # Some backends might not support this
        
        # Create grid layout
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 0: Rendered Map and Metadata
        ax = fig.add_subplot(gs[0, 0:2])
        if rendered_map.ndim == 3:
            ax.imshow(rendered_map.astype(np.uint8))
        else:
            im = ax.imshow(rendered_map, cmap='viridis', interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Rendered Map', fontsize=12, weight='bold')
        ax.axis('off')
        
        # Metadata display
        ax_meta = fig.add_subplot(gs[0, 2:4])
        ax_meta.axis('off')
        info_text = (
            f"Episode: {metadata['episode']}, Step: {metadata['step']}\n"
            f"Agent Position: ({metadata['agent_position'][0]:.1f}, {metadata['agent_position'][1]:.1f})\n"
            f"Agent Direction: {metadata['agent_direction']:.1f}°\n"
            f"Weed Count: {metadata['weed_count']}\n"
            f"Frontier Area: {metadata['frontier_area']}"
        )
        ax_meta.text(0.5, 0.5, info_text, ha='center', va='center',
                     fontsize=12, transform=ax_meta.transAxes,
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
        
        # Feature columns definition
        columns = [
            {
                'name': 'Frontier', 'col': 0, 'cmap': 'Blues',
                'full': features.get('frontier_full', np.zeros((1, 1))),
                'raw': features.get('frontier_raw', np.zeros((1, 1))),
                'apf': features.get('frontier_apf', np.zeros((1, 1)))
            },
            {
                'name': 'Weed', 'col': 1, 'cmap': 'Greens',
                'full': features.get('weed_full', np.zeros((1, 1))),
                'raw': features.get('weed_raw', np.zeros((1, 1))),
                'apf': features.get('weed_apf', np.zeros((1, 1)))
            },
            {
                'name': 'Obstacle', 'col': 2, 'cmap': 'Reds',
                'full': None,
                'raw': features.get('obstacle_raw', np.zeros((1, 1))),
                'apf': features.get('obstacle_apf', np.zeros((1, 1)))
            },
            {
                'name': 'Trajectory', 'col': 3, 'cmap': 'Purples',
                'full': None,
                'raw': features.get('trajectory_raw', np.zeros((1, 1))),
                'apf': features.get('trajectory_apf', features.get('trajectory_raw', np.zeros((1, 1))))
            }
        ]
        
        # Plot feature maps
        for col_data in columns:
            col = col_data['col']
            
            # Row 1: Full versions (or Mist for trajectory column)
            if col_data['full'] is not None:
                ax = fig.add_subplot(gs[1, col])
                im = ax.imshow(col_data['full'], cmap=col_data['cmap'], interpolation='nearest')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{col_data['name']} (Full)", fontsize=10)
                ax.axis('off')
            elif col == 3:  # Special case: Mist in trajectory column
                ax = fig.add_subplot(gs[1, col])
                mist_data = features.get('mist', np.zeros((1, 1)))
                im = ax.imshow(mist_data, cmap='gray', interpolation='nearest')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title("Mist", fontsize=10)
                ax.axis('off')
            
            # Row 2: Raw versions
            ax = fig.add_subplot(gs[2, col])
            im = ax.imshow(col_data['raw'], cmap=col_data['cmap'], interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{col_data['name']} (Raw)", fontsize=10)
            ax.axis('off')
            
            # Row 3: APF versions
            ax = fig.add_subplot(gs[3, col])
            im = ax.imshow(col_data['apf'], cmap=col_data['cmap'], interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{col_data['name']} (APF)", fontsize=10)
            ax.axis('off')
        
        # Add row labels
        row_labels = ['Full', 'Raw', 'APF']
        for i, label in enumerate(row_labels):
            fig.text(0.02, 0.75 - (i + 1) * 0.2, label, rotation=90, va='center',
                     fontsize=12, weight='bold', color='darkblue')
        
        plt.tight_layout()
        
        # Add helpful text for user interaction
        if not save_path:  # Only for interactive viewing
            fig.text(0.02, 0.02, 'Use toolbar to zoom/pan. Right-click to resize window.', 
                     fontsize=8, alpha=0.7, transform=fig.transFigure)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig
    
    def save_features(self, features):
        """Save features to disk with visualization and raw data."""
        frame_dir = self.feature_dir / f"frame_{self.frame_counter:06d}"
        frame_dir.mkdir(exist_ok=True)
        
        # Save visualization without displaying
        vis_path = frame_dir / "visualization.png"
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        
        fig = self.visualize_features(features, vis_path)
        plt.close(fig)
        matplotlib.use(backend)
        
        # Save raw data
        data_to_save = {}
        for key, value in features.items():
            if key == 'metadata':
                # Convert metadata to npz-compatible format
                data_to_save['metadata_step'] = value['step']
                data_to_save['metadata_episode'] = value['episode']
                data_to_save['metadata_agent_position'] = value['agent_position']
                data_to_save['metadata_agent_direction'] = value['agent_direction']
                data_to_save['metadata_weed_count'] = value['weed_count']
                data_to_save['metadata_frontier_area'] = value['frontier_area']
            elif isinstance(value, np.ndarray):
                data_to_save[key] = value
        
        np.savez_compressed(frame_dir / "features.npz", **data_to_save)
        print(f"Features saved to: {frame_dir}")
        self.frame_counter += 1
        
        return frame_dir
    
    def handle_events(self, paused):
        """Handle keyboard events and return new pause state."""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.fig:  # Close visualization window and resume
                        plt.close(self.fig)
                        self.fig = None
                        paused = False
                        print("Visualization closed, simulation resumed")
                    else:  # Normal pause/resume toggle
                        paused = not paused
                        print("Simulation " + ("paused" if paused else "resumed"))
                
                elif event.key == pygame.K_q and paused and self.config['log_feature']:
                    # Visualize current features
                    print("Visualizing features...")
                    if self.fig:
                        plt.close(self.fig)
                    features = self.env.get_current_features()
                    self.fig = self.visualize_features(features)
                    plt.show(block=False)
                    plt.draw()
                    plt.pause(0.001)
                    print("Press W to save, SPACE to close and resume")
                
                elif event.key == pygame.K_w and paused and self.config['log_feature']:
                    # Save current features
                    print("Saving features...")
                    features = self.env.get_current_features()
                    self.save_features(features)
                    print(f"Features saved! Total saved frames: {self.frame_counter}")
        
        return paused
    
    def wait_during_pause(self, paused):
        """Handle waiting during pause state."""
        if not paused:
            return
        
        if self.config['render']:
            self.env.render()
        
        # Handle matplotlib window responsiveness
        if self.fig:
            try:
                plt.pause(0.05)
            except:
                self.fig = None
        else:
            pygame.time.wait(50)
    
    def print_reward_details(self, t, reward, ret, info):
        """Print detailed reward information."""
        if 'reward_details' in info:
            details = info['reward_details']
            print(f'{t:04d} | Total: {reward:7.3f} (Σ={ret:7.3f}) | '
                  f'Const: {details["const"]:6.3f} | '
                  f'Turn: {details["turn"]:6.3f} | '
                  f'Frontier: {details["frontier"]:6.3f} | '
                  f'Weed: {details["weed"]:6.3f} | '
                  f'Extra: {details["extra"]:6.3f}')
        else:
            print(f'{t:04d} | {reward:.3f}, {ret:.3f}')
    
    def print_episode_summary(self, episode_idx, total_return):
        """Print episode summary with reward breakdown."""
        if self.config['log_reward'] and hasattr(self.env, 'get_episode_summary'):
            summary = self.env.get_episode_summary(-1)
            if summary:
                print(f"\nEpisode {episode_idx + 1} Summary:")
                print(f"  Total return: {total_return:.2f}")
                for key in ['const', 'turn', 'frontier', 'weed', 'extra']:
                    sum_key = f'{key}_sum'
                    mean_key = f'{key}_mean'
                    std_key = f'{key}_std'
                    print(f"  {key.capitalize():8s}: Sum={summary[sum_key]:7.2f}, "
                          f"Mean={summary[mean_key]:6.3f}, "
                          f"Std={summary[std_key]:6.3f}")
    
    def save_session_results(self):
        """Save overall session results."""
        # Save reward logs
        if self.config['log_reward'] and hasattr(self.env, 'save_rewards'):
            self.env.save_rewards(str(self.reward_dir / 'reward_details.csv'))
            self.env.plot_rewards(-1, save_path=str(self.reward_dir / 'episode_rewards.png'))
        
        # Print statistics
        print(f"\n=== Overall Statistics ===")
        print(f"Episodes completed: {len(self.episode_returns)}")
        if self.episode_returns:
            print(f"Average return: {np.mean(self.episode_returns):.2f} ± {np.std(self.episode_returns):.2f}")
            print(f"Min return: {np.min(self.episode_returns):.2f}")
            print(f"Max return: {np.max(self.episode_returns):.2f}")
        
        if self.config['log_feature']:
            print(f"\nTotal frames saved: {self.frame_counter}")
            print(f"Features saved to: {self.feature_dir}")
    
    def run(self):
        """Run the complete testing session."""
        print("=== SAC Test Session Started ===")
        print("Controls:")
        print("  SPACE: Pause/Resume (or close visualization and resume)")
        if self.config['log_feature']:
            print("  Q (when paused): Visualize current features")
            print("  W (when visualizing): Save features to disk")
            print("Visualization window features:")
            print("  - Toolbar: zoom, pan, home, back/forward navigation")
            print("  - Resizable: drag window edges to resize")
            print("  - Configurable size via 'visualization_figsize' in config")
        
        exploration_type = ExplorationType.RANDOM if self.config['act_randomly'] else ExplorationType.DETERMINISTIC
        
        with set_exploration_type(exploration_type), torch.no_grad():
            for i in range(self.config['episodes']):
                obs, info = self.env.reset(**self.config['env_reset_options'])
                done = False
                ret = 0.0
                t = 0
                paused = False
                
                print(f"\n=== Episode {i + 1}/{self.config['episodes']} ===")
                
                while not done:
                    # Handle events
                    paused = self.handle_events(paused)
                    
                    # Handle pause state
                    if paused:
                        self.wait_during_pause(paused)
                        continue
                    
                    # Process observation
                    if isinstance(obs, dict):
                        observation = obs['observation']
                        vector = obs['vector']
                    observation = torch.from_numpy(observation).float().to(self.device).unsqueeze(0)
                    vector = torch.tensor([vector]).float().to(self.device)
                    
                    # Get action from model
                    logits = self.actor(observation=observation, vector=vector)
                    action = logits[2][0].tolist()
                    
                    # Step environment
                    obs, reward, done, _, info = self.env.step(action)
                    t += 1
                    ret += reward
                    
                    # Print reward details
                    self.print_reward_details(t, reward, ret, info)
                    
                    # Render
                    if self.config['render']:
                        self.env.render()
                
                # Episode finished
                self.episode_returns.append(ret)
                self.print_episode_summary(i, ret)
        
        # Save results and cleanup
        self.save_session_results()
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.fig:
            plt.close(self.fig)
        self.env.close()
        pygame.quit()


def main():
    """Main function with configuration."""
    config = {
        # Core settings
        'episodes': 1,
        'render': True,
        'log_reward': False,
        'log_feature': False,
        'act_randomly': True,
        'device': 'cpu',
        
        # Visualization settings
        'visualization_figsize': (12, 9),  # Adjustable window size (width, height in inches)
        
        # Model path
        'model_path': '/home/lzh/NewCppRL/ckpt/sac_cont/0909/sac_our_model_con3_t[02350]_r[2703.08=2662.85~2782.18].pt',
        
        # Environment reset options
        'env_reset_options': {
            'seed': 120,  # Alternative: 88
            'options': {
                'weed_dist': 'gaussian',  # Alternative: 'uniform'
                'weed_num': 100,  # Alternative: 10
                # 'map_id': 66,  # Optional map selection
                # "specific_scenario_dir": real_map_dir  # Optional specific scenario
            }
        }
    }
    
    # Create and run test session
    session = SACTestSession(config)
    session.run()


if __name__ == "__main__":
    main()