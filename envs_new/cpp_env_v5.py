"""
CppEnv v5 - 田地覆盖任务 + HIF方向引导
"""
from __future__ import annotations

import math
import time

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from gymnasium.wrappers import HumanRendering

from envs_new.cpp_env_v4 import CppEnv as CppEnvV4
from envs_new.components.state.environment_state import EnvironmentState
from envs_new.components.config.environment_config import EnvironmentConfig
from envs_new.components.reward.reward_system import RewardCalculator
from envs_new.components.observation.observation_generator import ObservationGenerator
from envs_new.components.dynamics.environment_dynamics import Updater
from envs_new.components.render.renderer import Renderer
from envs_new.utils.image_utils import extract_ego_patch, apply_noise_to_pose


class CppEnv(CppEnvV4):
    """v5环境 - Field覆盖 + HIF方向引导"""

    def __init__(self, render_mode="rgb_array", **kwargs):
        v5_defaults = {'reward_hif': 0.01, }  # 'map_dir': "envs_new/maps/field_coverage" 默认HIF奖励权重
        final_kwargs = {**v5_defaults, **kwargs}

        super().__init__(render_mode=render_mode, **final_kwargs)

        # 添加HIF组件
        self.scenario_generator.add_component('hif', HIFCreator())
        self.reward_system.add_calculator('hif', HIFCalculator)
        self.env_dynamics.add_updater('ego_hif', EgoHIFUpdater(self.config))
        self.renderer = HIFRenderer(self.config)

        # 替换观测生成器为支持方向场的版本
        self.observation_generator = OrientationAwareObservationGenerator(self.config)

    def _get_observation_channels(self) -> int:
        """v5 通道数：v4基础（4或5） + HIF通道（2或3）"""
        base = super()._get_observation_channels()  # 4(field, obstalce, time_serverial_coverage, coverage_hot_map) + int(use_trajectory)
        hif_channels = 2 + int(self.config.include_confidence)  # cosine2, sine2, [confidence]
        return base + hif_channels

    def _get_observation_maps(self) -> Dict[str, Dict[str, Any]]:
        """v5 观测字典：v4基础 + observation_ego_hif（多通道复合负载）"""
        obs_maps = super()._get_observation_maps()  # 获取 v4 的基础观察地图
        if "observation_ego_hif" in self.maps_dict: # 添加 observation_ego_hif（显式声明通道数）
            hif_ch = 2 + int(self.config.include_confidence)  # cosine2, sine2, [confidence]
            obs_maps['observation_ego_hif'] = {'map': self.maps_dict["observation_ego_hif"], 'num_channels': hif_ch}  # 显式声明：2或3通道
        return obs_maps

    def set_pred_hif(self, pred_hif_dict: Dict[str, np.ndarray]) -> None:
        self.maps_dict['pred_ego_hif'] = pred_hif_dict # 注入预测的HIF场(cosine2, sine2, confidence)

class HIFCreator:
    """
    加载人类意图场(Human Intention Field)，用于引导agent的方向正则，HIF像素包描述方向值（弧度），表示该位置的期望行进方向，特殊值-1表示该位置无方向引导。
    """

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['field']  # HIF需要在field创建后加载，以获取地图尺寸

    def generate(self, state: Dict[str, Any], rng: np.random.Generator) -> None:
        """
        加载HIF地图并验证尺寸匹配， 场景模式：从scenario_directory加载，普通模式：从{map_dir}/hif/human_intent_field_{field_id}.npy加载
        """
        dimensions = state['env_state'].get_static_info('dimensions')
        scenario_directory = state['options']['scenario_directory']

        # 根据模式选择加载方式
        if scenario_directory:
            hif_map = self._load_from_directory(scenario_directory, dimensions)
        else:
            hif_map = self._load_from_file(state['env_state'].get_static_info('field_id'), state['config'], dimensions)

        # 若存在field缩放，则用同一中心仿射矩阵在向量域同步缩放HIF
        scale = float(state['env_state'].get_static_info('field_scale'))
        if abs(scale - 1.0) > 1e-6: hif_map = self._scale_hif_map_center(hif_map, scale)

        # 统一：转换到图像坐标系轴向角，并缓存向量域表示（保留无效为 -1）
        confidence = (hif_map >= 0.0).astype(np.float32)
        hif_image = np.where(confidence > 0.0, (np.pi - hif_map) % np.pi, -1.0).astype(np.float32)
        double_angles = 2.0 * np.where(confidence > 0.0, hif_image, 0.0)
        hif_cos2 = (np.cos(double_angles) * confidence).astype(np.float32)
        hif_sin2 = (np.sin(double_angles) * confidence).astype(np.float32)

        state['maps_dict']['hif'] = hif_image
        state['maps_dict']['hif_conf'] = confidence
        state['maps_dict']['hif_cos2'], state['maps_dict']['hif_sin2'] = hif_cos2, hif_sin2

    def _load_from_directory(self, directory: Union[str, Path], dimensions: Tuple[int, int]) -> np.ndarray:
        """从场景目录加载HIF地图"""
        hif_file = Path(directory) / 'map_hif.npy'
        return self._safe_load_map(hif_file, dimensions)

    def _load_from_file(self, field_id: Optional[int], config, dimensions: Tuple[int, int]) -> np.ndarray:
        """从标准位置加载HIF地图， config.map_dir指向包含field/和hif/的父目录"""
        map_root = Path(config.get_absolute_map_dir())
        hif_file = map_root / 'hif' / f'human_intent_field_{field_id}.npy'
        return self._safe_load_map(hif_file, dimensions)

    def _safe_load_map(self, file, dimensions: Tuple[int, int]) -> np.ndarray:
        """安全加载地图文件并验证尺寸"""
        if not file.exists(): raise FileNotFoundError(f"HIF file required but not found: {file}\n")
        hif_map = np.load(str(file)).astype(np.float32)
        if hif_map.shape != (dimensions[1], dimensions[0]): raise ValueError(
            f"HIF map dimensions {hif_map.shape} don't match expected {dimensions}")  # (height, width)
        return hif_map

    def _scale_hif_map_center(self, hif_map: np.ndarray, scale: float) -> np.ndarray:
        """以图像中心为原点对轴向HIF方向场做同心等比缩放（保持画布尺寸不变）。

        算法：向量场循环均值缩放
        1. 向量化：cos(2θ)·conf, sin(2θ)·conf（预乘confidence门控）
        2. 缩放：对(vx, vy, conf)使用面积插值
        3. 归一化：vx' = vx_scaled / conf_scaled（加权平均）
        4. 重建：θ' = 0.5·atan2(vy', vx')，仅在有效区域
        """

        h, w = hif_map.shape
        center_x, center_y = w / 2.0, h / 2.0
        transform_matrix = np.array([[scale, 0.0, (1.0 - scale) * center_x],
                                     [0.0, scale, (1.0 - scale) * center_y]], dtype=np.float32)

        # 转换为向量空间，并基于confidence门控自动清零-1区域
        confidence = (hif_map >= 0).astype(np.float32)
        double_angles = 2.0 * hif_map
        vx, vy = np.cos(double_angles) * confidence, np.sin(double_angles) * confidence

        # 向量场整体缩放（单次warpAffine）
        # INTER_AREA优化用于缩小（scale < 1）, 对于放大（scale > 1）, 考虑切换到INTER_LINEAR
        vector_field = np.stack([vx, vy, confidence], axis=-1)
        scaled_field = cv2.warpAffine(vector_field, transform_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                      flags=cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR)
        vx_scaled, vy_scaled, conf_scaled = scaled_field[..., 0], scaled_field[..., 1], scaled_field[..., 2]

        # 归一化重建平均向量
        vx_mean = vx_scaled / np.maximum(conf_scaled, 1e-6)
        vy_mean = vy_scaled / np.maximum(conf_scaled, 1e-6)

        # 有效性判断：足够覆盖率且向量幅值有效
        magnitude = np.hypot(vx_mean, vy_mean)
        valid = (conf_scaled > 0.1) & (magnitude > 1e-6)

        # 重建轴向角并规范到[0, π)
        out = np.full((h, w), -1.0, dtype=np.float32)
        if np.any(valid):
            angles_reconstructed = 0.5 * np.arctan2(vy_mean[valid], vx_mean[valid])
            out[valid] = np.mod(angles_reconstructed, np.pi)

        return out


class HIFCalculator(RewardCalculator):
    """
    人类意图场(HIF)方向引导奖励, 根据agent当前朝向与HIF指定方向的差异计算奖励, 角度差异越小，奖励越高（实际是负惩罚越小）
    """

    @classmethod
    def calculate(cls, env_state: EnvironmentState, coefficient: float,
                  config: EnvironmentConfig = None, **kwargs) -> float:
        """
        计算HIF方向引导奖励, 方向差异惩罚（负值）
        """
        # 获取方现场和agent位置信息
        hif_map = kwargs['map_dict']['hif']
        agent_position, agent_direction = env_state.get_info('agent_position'), env_state.get_info('agent_direction')

        # 获取agent当前和上一步的网格坐标
        x, y = int(agent_position.current[0]), int(agent_position.current[1])
        x_last, y_last = int(agent_position.last[0]), int(agent_position.last[1])

        # 计算与HIF方向差异
        weight_current, weight_last = 0.3, 0.7
        angle_diff = weight_current * cls._compute_angle_difference(agent_direction.current, hif_map[y, x]) \
            if hif_map[y, x] >= 0 else 0
        angle_diff_last = weight_last * cls._compute_angle_difference(agent_direction.current, hif_map[y_last, x_last]) \
            if hif_map[y_last, x_last] >= 0 else 0

        return -coefficient * (angle_diff + angle_diff_last)  # 返回负奖励（惩罚）

    @staticmethod
    def _compute_angle_difference(agent_direction: float, hif_direction: float) -> float:
        """
        计算无向（轴向）场中 Agent 朝向与 HIF 线方向的角度差（度）。

        坐标与单位：
        - Agent：图像坐标系（有向），0°=东，90°=南，顺时针递增，范围 [0, 360)，degree。
        - HIF：图像坐标系（无向轴向），0rad=东，范围 [0, π), radians。

        策略（轴向折叠）：
        - hif_axis_img_deg = degrees(hif_direction) % 180
        - delta = abs((agent_direction % 360) - hif_axis_img_deg) % 180
        - 若 delta > 90，则 delta = 180 - delta（轴向折叠），最终 delta ∈ [0, 90]
        """
        agent_direction_deg = float(agent_direction) % 360.0
        hif_axis_img_deg = float(np.degrees(hif_direction)) % 180.0
        # 计算图像系下的轴向差，并折叠到 [0, 90]
        delta_deg = abs(agent_direction_deg - hif_axis_img_deg) % 180.0
        if delta_deg > 90.0: delta_deg = 180.0 - delta_deg
        return float(delta_deg)


class EgoHIFUpdater(Updater):
    """生成自我中心坐标系下的HIF向量域observation and label ego hif patch（cos2/sin2/conf）。"""

    def __init__(self, config):
        self.config = config

    @classmethod
    def get_dependencies(cls) -> List[str]:
        return ['agent']

    def setup_state(self, state: Dict[str, Any], history_length: int = 2) -> None:
        """初始化HIF相关地图：observation_ego_hif（观测生成） + label_ego_hif（训练目标）"""
        h, w = int(self.config.state_downsize[0]), int(self.config.state_downsize[1])

        # observation_ego_hif: 预分配state_downsize尺寸用于观测生成
        state['maps_dict']['observation_ego_hif'] = {'cosine2': np.zeros((h, w), dtype=np.float32),
                                                     'sine2': np.zeros((h, w), dtype=np.float32),
                                                     'confidence': np.zeros((h, w), dtype=np.float32)}

        if self.config.use_multiscale: # label_ego_hif: 根据 use_multiscale 决定内存策略
            label_size = int(self.config.multiscale_feature_size) # 多尺度：独立分配 multiscale_feature_size 尺寸
            state['maps_dict']['label_ego_hif'] = {
                'cosine2': np.zeros((label_size, label_size), dtype=np.float32),
                'sine2': np.zeros((label_size, label_size), dtype=np.float32),
                'confidence': np.zeros((label_size, label_size), dtype=np.float32)}
        else:

            state['maps_dict']['label_ego_hif'] = state['maps_dict']['observation_ego_hif'] # 非多尺度，state_downsize尺寸

    def update(self, state: Dict[str, Any]) -> None:
        """更新HIF地图：observation_ego_hif（观测） + label_ego_hif（训练目标）"""
        maps, agent = state['maps_dict'], state['agent']

        # 需要全局HIF向量域缓存
        if not all(k in maps for k in ('hif_cos2', 'hif_sin2', 'hif_conf')):
            return

        # 提取并旋转到自我坐标系（固定 768×768）
        h, w = int(self.config.state_downsize[0]), int(self.config.state_downsize[1])
        stacked = np.stack([maps['hif_cos2'], maps['hif_sin2'], maps['hif_conf']], axis=2)
        ego_patch = extract_ego_patch(maps=stacked, pad_values=[0.0, 0.0, 0.0], patch_size=(h, w), center_y=agent.y,
                                     center_x=agent.x, direction_deg=agent.direction).astype(np.float32)

        ego_cos, ego_sin, ego_conf = ego_patch[..., 0], ego_patch[..., 1], ego_patch[..., 2]

        # 双倍角旋转补偿到自我坐标（EGO_ROTATION_OFFSET=90.0）
        rotation_angle = -2.0 * math.radians(90.0 + float(agent.direction))
        cos_r, sin_r = math.cos(rotation_angle), math.sin(rotation_angle)
        cos_rel = cos_r * ego_cos - sin_r * ego_sin
        sin_rel = sin_r * ego_cos + cos_r * ego_sin

        # 原地更新 observation_ego_hif (768×768)
        obs = maps['observation_ego_hif']
        obs['cosine2'][...], obs['sine2'][...], obs['confidence'][...] = cos_rel, sin_rel, ego_conf

        # 原地更新 label_ego_hif
        if self.config.use_multiscale:
            label_size = int(self.config.multiscale_feature_size)
            cy, cx, half = h // 2, w // 2, label_size // 2
            maps['label_ego_hif']['cosine2'][...] = cos_rel[cy-half:cy+half, cx-half:cx+half]
            maps['label_ego_hif']['sine2'][...] = sin_rel[cy-half:cy+half, cx-half:cx+half]
            maps['label_ego_hif']['confidence'][...] = ego_conf[cy-half:cy+half, cx-half:cx+half]
        else:
            state['maps_dict']['label_ego_hif'] = state['maps_dict']['observation_ego_hif']


class HIFRenderer(Renderer):
    """在父类渲染基础上叠加ego HIF方向线段（仅map模式）。"""

    def render(self, maps_dict: Dict[str, np.ndarray], agent, dimensions: Tuple[int, int],
               mode: str = "map", observation_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        base = super().render(maps_dict, agent, dimensions, mode, observation_size)

        if mode != 'map' or not self.config.render_hif_lines: return base
        if 'label_ego_hif' not in maps_dict: return base  # 策略：始终渲染双面板（GT | Pred），无 pred 时用黑色占位符确保 observation_spec 恒定  # 无HIF数据，返回基础渲染

        # 左侧：GT HIF（始终存在）
        label_panel = self._overlay_hif_lines(base.copy(), maps_dict['label_ego_hif'], maps_dict, agent)

        # 右侧：Pred HIF（可能不存在，用占位符填充）
        pred_panel = np.zeros_like(label_panel)  if 'pred_ego_hif'  not in maps_dict else ( # 黑色占位符
            self._overlay_hif_lines(base.copy(), maps_dict['pred_ego_hif'], maps_dict, agent)) # 预测HIF
        return np.hstack([label_panel, pred_panel])  # 始终返回 [H, 2W, 3]

    def _overlay_hif_lines(self, image: np.ndarray, hif_dict: Dict[str, np.ndarray],
                          maps_dict: Dict[str, np.ndarray], agent) -> np.ndarray:
        """纯函数风格的HIF线段叠加. image: 基础渲染图像, hif_dict: HIF数据字典，包含'cosine2', 'sine2', 'confidence'键, maps_dict: 完整地图字典（用于边界检查）, agent: 智能体对象
        """
        ego_cos, ego_sin, ego_conf = hif_dict['cosine2'], hif_dict['sine2'], hif_dict['confidence']
        height, width = ego_cos.shape

        # 1. 计算ego patch尺寸和缩放比例
        if self.config.use_multiscale:
            patch_h = patch_w = int(self.config.multiscale_feature_size)
        else:
            patch_h, patch_w = int(self.config.state_downsize[0]), int(self.config.state_downsize[1])

        patch_center_x, patch_center_y = width / 2.0, height / 2.0
        scale_to_patch_x, scale_to_patch_y = patch_w / max(width, 1), patch_h / max(height, 1)
        render_scale = self.config.render_repeat_times

        # 2. 预计算ego→world坐标旋转矩阵（extract_ego_patch使用90°+direction）
        ego_rotation_deg = agent.direction + 90.0
        ego_to_world_rad = math.radians(ego_rotation_deg)
        cos_ego_world, sin_ego_world = math.cos(ego_to_world_rad), math.sin(ego_to_world_rad)

        # 3. 预计算方向向量的double-angle旋转（轴向编码需要双倍角）
        double_angle_rad = 2.0 * ego_to_world_rad
        cos_double, sin_double = math.cos(double_angle_rad), math.sin(double_angle_rad)

        # 4. 预计算渲染参数
        line_half_length = 0.5 * self.config.hif_line_length * render_scale
        line_thickness = max(1, int(self.config.hif_line_thickness * render_scale))

        # 5. 遍历HIF网格，绘制方向线段
        for iy in range(0, height, self.config.hif_line_stride):
            for ix in range(0, width, self.config.hif_line_stride):
                if ego_conf[iy, ix] < self.config.hif_line_confidence_threshold: continue
                # ego patch坐标（原点在中心）→ 缩放到观测尺寸
                offset_x, offset_y = ix - patch_center_x, iy - patch_center_y
                dx_ego, dy_ego = offset_x * scale_to_patch_x, offset_y * scale_to_patch_y

                # ego坐标 → world坐标（刚体变换：旋转+平移）
                dx_world = cos_ego_world * dx_ego - sin_ego_world * dy_ego
                dy_world = sin_ego_world * dx_ego + cos_ego_world * dy_ego

                # world坐标 → 渲染画布坐标（应用render_scale）
                x_render = int(round((agent.x + dx_world) * render_scale))
                y_render = int(round((agent.y + dy_world) * render_scale))

                if not (0 <= x_render < image.shape[1] and 0 <= y_render < image.shape[0]): continue

                # 检查该点是否在有效田地内（使用原始地图尺寸）
                x_world_orig, y_world_orig = int(round(agent.x + dx_world)), int(round(agent.y + dy_world))
                if maps_dict['original_field'][y_world_orig, x_world_orig] == 0: continue

                # ego方向向量 → world方向向量（double-angle旋转）
                cos2_ego, sin2_ego = ego_cos[iy, ix], ego_sin[iy, ix]
                cos2_world = cos_double * cos2_ego - sin_double * sin2_ego
                sin2_world = sin_double * cos2_ego + cos_double * sin2_ego
                orientation_world = 0.5 * math.atan2(sin2_world, cos2_world)

                # 计算线段端点并绘制
                dx, dy = line_half_length * math.cos(orientation_world), line_half_length * math.sin(orientation_world)
                pt1 = (int(round(x_render - dx)), int(round(y_render - dy)))
                pt2 = (int(round(x_render + dx)), int(round(y_render + dy)))
                cv2.line(image, pt1, pt2, self.config.hif_line_color, line_thickness)
        return image


class OrientationAwareObservationGenerator(ObservationGenerator):
    """
    专门处理方向场(HIF)的观测生成器，
    核心思想：将无向循环角度场转换为适合神经网络的向量表示 （1）使用双倍角编码解决轴向等价性(θ与θ+π等价) （2）转换到自我中心坐标系 （3）使用加权平均池化保持方向信息
    """
    EGO_ROTATION_OFFSET = 90.0  # 自我中心视角旋转偏移(度)

    def __init__(self, config):
        super().__init__(config)

    def generate_observation(self, agent, obs_maps, noisy_pose=None):
        """生成观测：使用 observation_ego_hif (768×768) 支持完整多尺度金字塔"""
        noisy_pose = noisy_pose if noisy_pose is not None else apply_noise_to_pose(agent.y, agent.x, agent.direction,
                                                                                   self.config.position_noise,
                                                                                   self.config.direction_noise,
                                                                                   self.rng)
        if 'observation_ego_hif' in obs_maps: # 观测中存在observation_ego_hif，调用单独的循环方现场观测处理分支
            hif_obs = self._process_ego_hif_label(obs_maps.pop('observation_ego_hif')['map'])
            base_obs = super().generate_observation(agent, obs_maps, noisy_pose)
            return np.concatenate([base_obs, hif_obs], axis=0)
        else:
            return super().generate_observation(agent, obs_maps, noisy_pose) # 若观测不存在observation_ego_hif，退化为普通观测生成方法

    def _process_ego_hif_label(self, hif_label: Dict[str, np.ndarray]) -> np.ndarray:
        """处理observation_ego_hif，转换为适合神经网络的观测格式。
        输入: hif_label = {'cosine2': [...], 'sine2': [...], 'confidence': [...]}，值域: cos2/sin2 ∈ [-1, 1], confidence ∈ [0, 1]
        输出: (C, H, W) 观测，归一化到[0, 1]
        """
        ego_cos, ego_sin = hif_label['cosine2'].astype(np.float32), hif_label['sine2'].astype(np.float32)
        ego_conf = hif_label['confidence'].astype(np.float32)
        normalized_cos, normalized_sin = 0.5 + 0.5 * ego_cos, 0.5 + 0.5 * ego_sin # cos/sin从[-1,1]映射到[0,1]

        # 过滤无效置信域区域
        mask_invalid = (ego_conf <= 1e-6)
        normalized_cos[mask_invalid], normalized_sin[mask_invalid] = 0.0, 0.0

        # 组装观测通道
        hif_channels = [normalized_cos, normalized_sin]
        if self.config.include_confidence:
            hif_channels.append(np.clip(ego_conf, 0.0, 1.0))
        hif_obs = np.stack(hif_channels, axis=0).astype(np.float32)

        if self.config.use_multiscale: # 多尺度处理
            return self._apply_orientation_multiscale(hif_obs)
        return hif_obs

    def _apply_orientation_multiscale(self, ego_hif_patch: np.ndarray) -> np.ndarray:
        """方向场专用的多尺度处理, 使用加权平均而非最大池化"""
        feature_size = self.config.multiscale_feature_size
        center_size = ego_hif_patch.shape[-1] // 2

        ego_hif_patch = torch.from_numpy(ego_hif_patch).unsqueeze(0)  ## 转换为PyTorch张量, (1, C, H, W)
        original_hif_patch = ego_hif_patch.clone() if self.config.use_global_features else None  # 保存原始张量用于全局特征提取

        multiscale_features = []
        with torch.no_grad():
            for _ in range(self.config.n_scales):  # 生成 n_scales 个尺度hif观测
                half_size = feature_size // 2
                cropped = ego_hif_patch[0, :, center_size - half_size:center_size + half_size,
                          center_size - half_size:center_size + half_size]
                multiscale_features.append(cropped.numpy())

                # 加权池化到下一尺度
                ego_hif_patch = self._weighted_avg_pool(ego_hif_patch, kernel_size=2)
                center_size //= 2

            if self.config.use_global_features:
                kernel_size = int(np.round(self.config.state_downsize[0] / feature_size))  # 计算池化核大小
                assert kernel_size >= 1, f"Invalid kernel_size: {kernel_size}"  # 这些检查应该在初始化时就通过了，这里用断言确保
                multiscale_features.append(self._weighted_avg_pool(original_hif_patch, kernel_size)[0].numpy())

        return np.concatenate(multiscale_features, axis=0).astype(np.float32)

    def _weighted_avg_pool(self, tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """统一的加权平均池化函数"""
        if self.config.include_confidence:
            value, weights = tensor[:, :2], tensor[:, 2:3]  # 分离方向向量和置信度

            # 加权平均池化
            weighted_pooled_value = F.avg_pool2d(value * weights, kernel_size, kernel_size)  # Σ(v*w)/n
            pooled_weights = F.avg_pool2d(weights, kernel_size, kernel_size)  # 只池化一次 # Σw/n
            normalized_pooled_value = weighted_pooled_value / pooled_weights.clamp_min(
                self.config.hif_min_epsilon)  # Σ(v*w)/Σw(n约分掉了)

            return torch.cat([normalized_pooled_value, pooled_weights], dim=1)
        else:  # 无置信度时直接池化
            return F.avg_pool2d(tensor, kernel_size, kernel_size)

if __name__ == "__main__":
    if_render = True
    episodes = 3

    print("=" * 60)
    print("Testing CppEnv v5 - Field Coverage + HIF Guidance")
    print("=" * 60)

    # 创建v5环境，展示HIF特性
    env = CppEnv(use_multiscale=True, use_global_features=True,
                 field_scale_enabled=False,
                 field_scale_range=(0.5, 0.7), # (1.0, 1.0)
                 # render_first_person=True,  # 控制渲染第一人称视角
                 )
    # 默认map_dir现在指向field_coverage，自动寻找field/和hif/子目录

    if if_render: env = HumanRendering(env)

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")

        # 使用特定地图ID以加载HIF（如果有的话）
        # env.update_config({'num_obstacles_range':[0,0]})
        obs, info = env.reset(seed=120 + episode, options={
            # 'map_id': 0,  # 使用地图0（如果存在对应的HIF文件）
            # 'initial_position': None,
            # 'initial_direction': None,
        })

        print(f"Observation shape: {obs['observation'].shape}")
        print(f"Initial field coverage: {obs['completion_ratio'][0]:.2%}")

        # 检查是否成功加载HIF（通过observation判断）
        if 'observation' in obs and obs['observation'].shape[0] > 3:
            print(f"HIF observation channel detected (total channels: {obs['observation'].shape[0]})")

        env.action_space.seed(66)
        done = False
        step_count = 0
        total_reward = 0
        hif_rewards = []

        while not done:  # 限制步数用于测试
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            step_count += 1
            total_reward += reward

            # 尝试分解奖励以查看HIF贡献
            reward_breakdown = env.reward_system.get_reward_breakdown(
                env.env_state, map_dict=env.maps_dict)
            if 'hif' in reward_breakdown['breakdown']:
                hif_rewards.append(reward_breakdown['breakdown']['hif'])

            if step_count % 20 == 0:
                print(f"  Step {step_count}: reward={reward:.4f}, "
                      f"coverage={obs['completion_ratio'][0]:.2%}")
                if hif_rewards:
                    print(f"    HIF reward contribution: {hif_rewards[-1]:.4f}")

            if if_render:
                env.render()

            if done or truncated:
                print(f"\nEpisode finished after {step_count} steps!")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Final coverage: {obs['completion_ratio'][0]:.2%}")
                if hif_rewards:
                    avg_hif = np.mean(hif_rewards)
                    print(f"  Average HIF reward: {avg_hif:.4f}")
                if info.get('crashed'):
                    print("  Termination: Crashed")
                elif info.get('finished'):
                    print("  Termination: Field fully covered!")
                elif info.get('timeout'):
                    print("  Termination: Timeout")
                break

    env.close()
    print("\n✅ v5 Environment test completed successfully!")
