import gymnasium as gym
from envs.cpp_env_v2 import CppEnv
from gymnasium.wrappers import HumanRendering


def get_env():
    render = True
    env = gym.make(id="Pasture-v2", render_mode='rgb_array' if render else None, action_type="continuous", state_size=(128, 128), state_downsize=(128, 128), num_obstacles_range=(5, 8), use_sgcnn=True, use_global_obs=True, use_apf=True, use_box_boundary=True, use_traj=True, noise_position=0, noise_direction=0, noise_weed=0)

    if render:
        env = HumanRendering(env)  # noqa
    if render:
        env.render()

    obs, info = env.reset(seed=47, options={'weed_dist': 'uniform', 'map_id': 63, 'weed_num': 200})
    return env, obs

