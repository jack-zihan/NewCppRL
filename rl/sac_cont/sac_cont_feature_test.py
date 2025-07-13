from pathlib import Path
import gymnasium as gym
import torch
import yaml
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib
from gymnasium.wrappers import HumanRendering
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type
from datetime import datetime

import envs  # noqa
from envs.wrapper.reward_tracker import RewardTracker
from envs.wrapper.feature_tracker import FeatureTracker

# Set matplotlib backend to TkAgg for better window handling
matplotlib.use('TkAgg')
plt.ion()  # 启用交互模式，确保窗口稳定显示

# Initialize pygame for keyboard eventsget_current_features
pygame.init()

# Configuration
base_dir = Path(__file__).parent.parent.parent
cfg = DictConfig(yaml.load(open(f'{base_dir}/configs/env_config.yaml'), Loader=yaml.FullLoader))
episodes = 1
render = True
log_reward = False
log_feature = False  # New flag for feature tracking
act_randomly = True
# act_randomly = False

# Feature saving directory
feature_save_dir = base_dir / 'logs' / 'feature_analysis' / datetime.now().strftime('%Y%m%d_%H%M%S')
if log_feature:
    feature_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Features will be saved to: {feature_save_dir}")

# Load model
device = 'cpu'
pt_path = f'/home/lzh/NewCppRL/ckpt/sac_cont/0909/sac_our_model_con3_t[02350]_r[2703.08=2662.85~2782.18].pt'
actor_critic = torch.load(pt_path).to(device)
actor = actor_critic[0].to(device)

# Create environment
env = gym.make(
    render_mode='rgb_array' if render else None,
    **cfg.env.params,
)

# Apply wrappers
if log_reward:
    env = RewardTracker(env)

if log_feature:
    env = FeatureTracker(env)

if render:
    env = HumanRendering(env)

exploration_type = ExplorationType.RANDOM if act_randomly else ExplorationType.DETERMINISTIC
episode_returns = []


def visualize_features(features, save_path=None):
    """Visualize all captured features in a single figure."""
    # Extract features
    rendered_map = features['rendered_map']
    metadata = features['metadata']

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Environment Features - Episode {metadata['episode']}, Step {metadata['step']}", fontsize=16)

    # Use GridSpec for flexible layout
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Row 0: Rendered Map and Metadata
    ax = fig.add_subplot(gs[0, 0:2])  # Rendered Map takes 2 columns
    if rendered_map.ndim == 3:
        ax.imshow(rendered_map.astype(np.uint8))
    else:
        im = ax.imshow(rendered_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Rendered Map', fontsize=12, weight='bold')
    ax.axis('off')

    # Metadata info in the right half of first row
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

    # Define columns: Frontier, Weed, Obstacle, Trajectory
    columns = [
        {
            'name': 'Frontier',
            'col': 0,
            'cmap': 'Blues',
            'full': features.get('frontier_full', np.zeros((1, 1))),
            'raw': features.get('frontier_raw', np.zeros((1, 1))),
            'apf': features.get('frontier_apf', np.zeros((1, 1)))
        },
        {
            'name': 'Weed',
            'col': 1,
            'cmap': 'Greens',
            'full': features.get('weed_full', np.zeros((1, 1))),
            'raw': features.get('weed_raw', np.zeros((1, 1))),
            'apf': features.get('weed_apf', np.zeros((1, 1)))
        },
        {
            'name': 'Obstacle',
            'col': 2,
            'cmap': 'Reds',
            'full': None,  # Obstacle doesn't have full version
            'raw': features.get('obstacle_raw', np.zeros((1, 1))),
            'apf': features.get('obstacle_apf', np.zeros((1, 1)))
        },
        {
            'name': 'Trajectory',
            'col': 3,
            'cmap': 'Purples',
            'full': None,  # Trajectory doesn't have full version, use mist instead
            'raw': features.get('trajectory_raw', np.zeros((1, 1))),
            'apf': features.get('trajectory_apf', features.get('trajectory_raw', np.zeros((1, 1))))
            # Fallback to raw if apf not available
        }
    ]

    # Row labels
    row_labels = ['Full', 'Raw', 'APF']

    # Plot all feature maps
    for col_data in columns:
        col = col_data['col']

        # Row 1: Full versions
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

    # Add row labels on the left
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.75 - (i + 1) * 0.2, label, rotation=90, va='center',
                 fontsize=14, weight='bold', color='darkblue')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    return fig

# Feature saving function
def save_features(features, save_dir, frame_id):
    """Save all features to disk."""
    frame_dir = save_dir / f"frame_{frame_id:06d}"
    frame_dir.mkdir(exist_ok=True)

    # Save visualization (without displaying)
    vis_path = frame_dir / "visualization.png"
    # Create figure without showing it
    import matplotlib
    backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Use non-interactive backend temporarily

    fig = visualize_features(features, vis_path)
    plt.close(fig)  # Close the figure immediately

    matplotlib.use(backend)  # Restore original backend

    # Save raw data
    data_to_save = {}
    for key, value in features.items():
        if key == 'metadata':
            # Save metadata as npz-compatible dict
            data_to_save['metadata_step'] = value['step']
            data_to_save['metadata_episode'] = value['episode']
            data_to_save['metadata_agent_position'] = value['agent_position']
            data_to_save['metadata_agent_direction'] = value['agent_direction']
            data_to_save['metadata_weed_count'] = value['weed_count']
            data_to_save['metadata_frontier_area'] = value['frontier_area']
        elif isinstance(value, np.ndarray):
            data_to_save[key] = value

    # Save as npz file
    np.savez_compressed(frame_dir / "features.npz", **data_to_save)
    print(f"Features saved to: {frame_dir}")

    return frame_dir


# Main loop
frame_counter = 0
with set_exploration_type(exploration_type), torch.no_grad():
    for i in range(episodes):
        obs, info = env.reset(seed=120,
                              options={
                                  'weed_dist': 'gaussian',
                                  "weed_num": 100,
                              })
        done = False
        ret = 0.
        t = 0

        print(f"\n=== Episode {i + 1}/{episodes} ===")
        if log_feature:
            print("Controls:")
            print("  SPACE: Pause/Resume (or close visualization and resume)")
            print("  Q (when paused): Visualize current features")
            print("  W (when visualizing): Save features to disk")

        paused = False
        fig = None  # Current matplotlib figure

        while not done:
            # Check keyboard events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if fig:  # 如果有可视化窗口打开
                            # 关闭窗口并继续播放
                            plt.close(fig)
                            fig = None
                            paused = False
                            print("Visualization closed, simulation resumed")
                        else:
                            # 正常的暂停/继续切换
                            paused = not paused
                            print("Simulation " + ("paused" if paused else "resumed"))

                    elif event.key == pygame.K_q and paused and log_feature:
                        # Visualize features (only visualization, no saving)
                        print("Visualizing features...")
                        if fig:
                            plt.close(fig)
                        features = env.get_current_features()
                        fig = visualize_features(features)  # 不传save_path参数
                        plt.show(block=False)
                        plt.draw()
                        plt.pause(0.001)
                        print("Press W to save, SPACE to close without saving and resume")
                    elif event.key == pygame.K_w and paused and log_feature:
                        # Save features (without re-visualizing)
                        print("Saving features...")
                        features = env.get_current_features()
                        save_dir = save_features(features, feature_save_dir, frame_counter)
                        frame_counter += 1
                        print(f"Features saved to {save_dir}! Total saved frames: {frame_counter}")
                        # 保存后不关闭可视化窗口，用户可以继续查看
            # If paused, continue rendering but don't step
            if paused:
                if render:
                    env.render()

                # 添加matplotlib事件处理，保持窗口响应
                if fig:
                    try:
                        plt.pause(0.05)  # 使用plt.pause保持窗口响应
                    except:
                        # 如果窗口被关闭，清理fig引用
                        fig = None
                else:
                    pygame.time.wait(50)  # 只有在没有matplotlib窗口时才使用pygame等待

                continue

            # Normal simulation step
            if isinstance(obs, dict):
                observation = obs['observation']
                vector = obs['vector']
            observation = torch.from_numpy(observation).float().to(device).unsqueeze(0)
            vector = torch.tensor([vector]).float().to(device).unsqueeze(0)

            # Get action from model
            logits = actor(observation=observation, vector=vector)
            action = logits[2][0].tolist()
            obs, reward, done, _, info = env.step(action)
            t += 1
            ret += reward

            # Print reward details if available
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

            if render:
                env.render()

        episode_returns.append(ret)

        # Episode summary
        if log_reward and hasattr(env, 'get_episode_summary'):
            summary = env.get_episode_summary(-1)
            if summary:
                print(f"\nEpisode {i + 1} Summary:")
                print(f"  Total return: {ret:.2f}")
                for key in ['const', 'turn', 'frontier', 'weed', 'extra']:
                    sum_key = f'{key}_sum'
                    mean_key = f'{key}_mean'
                    std_key = f'{key}_std'
                    print(f"  {key.capitalize():8s}: Sum={summary[sum_key]:7.2f}, "
                          f"Mean={summary[mean_key]:6.3f}, "
                          f"Std={summary[std_key]:6.3f}")

# Save reward logs if enabled
if log_reward and hasattr(env, 'save_rewards'):
    output_dir = Path(f'{base_dir}/logs/reward_analysis')
    output_dir.mkdir(exist_ok=True)
    env.save_rewards(str(output_dir / 'reward_details.csv'))
    env.plot_rewards(-1, save_path=str(output_dir / 'episode_rewards.png'))

# Print overall statistics
print(f"\n=== Overall Statistics ===")
print(f"Episodes completed: {episodes}")
print(f"Average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
print(f"Min return: {np.min(episode_returns):.2f}")
print(f"Max return: {np.max(episode_returns):.2f}")

if log_feature:
    print(f"\nTotal frames saved: {frame_counter}")
    print(f"Features saved to: {feature_save_dir}")

env.close()
pygame.quit()