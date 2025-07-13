"""
2025-07-12
v1-v2-v3-v4-v5-v6-v7-v8: 实现小车的追踪、光流矫正、实时矫正、轨迹和标注渲染等功能
想到同时考虑顺序以及微调太困难，因此拆解为visualize追踪标注，refine顺序渲染微调优化的架构
"""

import cv2
import numpy as np
from enum import Enum
from collections import deque
import time
import os
import torch
import threading
import queue
from abc import ABC, abstractmethod
import concurrent.futures

# 添加最终视频“发现”标注的refine操作

# 检查GPU可用性
GPU_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
if GPU_AVAILABLE:
    print("CUDA加速可用")
else:
    print("CUDA加速不可用，将使用CPU")

# 检查PyTorch GPU可用性
TORCH_GPU_AVAILABLE = torch.cuda.is_available()
if TORCH_GPU_AVAILABLE:
    print("PyTorch GPU加速可用")
    DEVICE = torch.device('cuda')
else:
    print("PyTorch GPU加速不可用，将使用CPU")
    DEVICE = torch.device('cpu')


# 定义程序状态
class State(Enum):
    INITIALIZING = 0  # 新增初始化状态
    PLAYING = 1
    PAUSED = 2
    DRAWING_POLYGON = 3
    DRAWING_CIRCLE = 4


# ===== 特征提取器接口 =====
class FeatureExtractorInterface(ABC):
    """特征提取器接口"""

    @abstractmethod
    def extract_features(self, image):
        """提取图像特征"""
        pass

    @abstractmethod
    def match_features(self, features1, features2):
        """匹配两组特征"""
        pass


# ===== SIFT特征提取器 =====
class SIFTFeatureExtractor(FeatureExtractorInterface):
    """SIFT特征提取器 - 复用SIFT检测器"""

    def __init__(self, config):
        """初始化SIFT特征提取器

        Args:
            config (dict): 配置参数
        """
        # 创建一次SIFT检测器并在整个生命周期内复用
        self.sift = cv2.SIFT_create(
            nfeatures=config.get('sift_features', 500),
            contrastThreshold=config.get('sift_contrast', 0.04),
            edgeThreshold=config.get('sift_edge', 10)
        )

        # 初始化FLANN特征匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # 匹配参数
        self.match_ratio = config.get('match_ratio', 0.7)
        self.debug = config.get('debug_feature_matching', False)

    def extract_features(self, image):
        """提取SIFT特征

        Args:
            image: 输入图像

        Returns:
            tuple: (关键点, 描述符)
        """
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 提取特征
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        if self.debug and keypoints is not None:
            print(f"提取了 {len(keypoints)} 个SIFT特征点")

        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """匹配两组特征描述符

        Args:
            desc1: 第一组描述符
            desc2: 第二组描述符

        Returns:
            list: 匹配列表
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []

        # 获取kNN匹配（k=2）
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # 应用比率测试筛选好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)

        if self.debug:
            print(f"找到 {len(good_matches)} 个好的匹配")

        return good_matches


# ===== CNN特征提取器 =====
class CNNFeatureExtractor(FeatureExtractorInterface):
    """使用CNN模型提取更稳健的特征"""

    def __init__(self, config):
        """初始化CNN特征提取器

        Args:
            config (dict): 配置参数
        """
        self.config = config
        self.model_name = config.get('cnn_model', 'mobilenet_v2')
        self.debug = config.get('debug_feature_matching', False)

        # 创建CNN模型的标志
        self.model_loaded = False

        # 初始化SIFT作为备选
        self.sift = cv2.SIFT_create(
            nfeatures=config.get('sift_features', 500),
            contrastThreshold=config.get('sift_contrast', 0.04),
            edgeThreshold=config.get('sift_edge', 10)
        )

        # 初始化FLANN特征匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.match_ratio = config.get('match_ratio', 0.7)

        # 尝试加载深度学习模型
        self.load_model()

    def load_model(self):
        """加载深度学习模型"""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            # 设置转换
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # 加载预训练模型
            if self.model_name == 'mobilenet_v2':
                self.model = models.mobilenet_v2(pretrained=True)
                # 移除分类层，仅保留特征提取部分
                self.model.classifier = torch.nn.Identity()
            elif self.model_name == 'resnet18':
                self.model = models.resnet18(pretrained=True)
                self.model.fc = torch.nn.Identity()
            else:
                print(f"不支持的模型类型: {self.model_name}，将使用SIFT特征")
                return

            # 将模型移动到设备
            self.model = self.model.to(DEVICE)
            self.model.eval()

            self.model_loaded = True
            print(f"成功加载 {self.model_name} 深度学习模型")

        except Exception as e:
            print(f"加载深度学习模型失败: {e}")
            print("将使用SIFT特征作为备选")
            self.model_loaded = False

    def extract_features(self, image):
        """提取CNN特征和SIFT特征

        Args:
            image: 输入图像

        Returns:
            dict: 包含CNN特征和SIFT特征的字典
        """
        # 提取SIFT特征（用于视觉匹配）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, sift_desc = self.sift.detectAndCompute(gray, None)

        # 如果CNN模型已加载，提取CNN特征
        cnn_features = None
        if self.model_loaded:
            try:
                import torch
                from PIL import Image

                # 预处理图像
                input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)

                # 提取特征
                with torch.no_grad():
                    cnn_features = self.model(input_tensor)
                    cnn_features = cnn_features.squeeze().cpu().numpy()

                if self.debug:
                    print(f"提取CNN特征成功，形状: {cnn_features.shape}")
            except Exception as e:
                print(f"CNN特征提取失败: {e}")
                cnn_features = None

        return {
            'keypoints': keypoints,
            'sift_descriptors': sift_desc,
            'cnn_features': cnn_features
        }

    def match_features(self, features1, features2):
        """匹配两组特征

        Args:
            features1: 第一组特征字典
            features2: 第二组特征字典

        Returns:
            dict: 匹配结果
        """
        # 首先尝试CNN特征匹配
        similarity = None
        if (features1.get('cnn_features') is not None and
                features2.get('cnn_features') is not None and
                self.model_loaded):
            try:
                # 计算余弦相似度
                f1 = features1['cnn_features']
                f2 = features2['cnn_features']

                # 归一化特征向量
                f1_norm = f1 / np.linalg.norm(f1)
                f2_norm = f2 / np.linalg.norm(f2)

                # 计算相似度
                similarity = np.dot(f1_norm, f2_norm)

                if self.debug:
                    print(f"CNN特征相似度: {similarity:.4f}")
            except Exception as e:
                print(f"CNN特征匹配失败: {e}")
                similarity = None

        # 然后执行SIFT特征匹配
        sift_matches = []
        if (features1.get('sift_descriptors') is not None and
                features2.get('sift_descriptors') is not None):

            desc1 = features1['sift_descriptors']
            desc2 = features2['sift_descriptors']

            if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
                # 获取kNN匹配（k=2）
                matches = self.matcher.knnMatch(desc1, desc2, k=2)

                # 应用比率测试
                for m, n in matches:
                    if m.distance < self.match_ratio * n.distance:
                        sift_matches.append(m)

                if self.debug:
                    print(f"找到 {len(sift_matches)} 个SIFT匹配")

        return {
            'cnn_similarity': similarity,
            'sift_matches': sift_matches
        }


# ===== 深度估计器 =====
class DepthEstimator:
    """深度图估计器"""

    def __init__(self, config):
        """初始化深度估计器

        Args:
            config (dict): 配置参数
        """
        self.config = config
        self.model_type = config.get('depth_model_type', 'MiDaS_small')
        self.depth_available = False

        # 尝试加载深度估计模型
        self.load_model()

    def load_model(self):
        """加载MiDaS深度估计模型"""
        try:
            import torch

            # 尝试加载MiDaS模型
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(DEVICE)
            self.model.eval()

            # 获取相应的转换
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self.depth_available = True
            print(f"成功加载 {self.model_type} 深度估计模型")

        except Exception as e:
            print(f"加载深度估计模型失败: {e}")
            print("深度估计功能不可用，将使用光流法估计相机运动")
            self.depth_available = False

    def estimate_depth(self, frame):
        """估计输入帧的深度图

        Args:
            frame: 输入图像帧

        Returns:
            ndarray: 深度图，如果不可用则返回None
        """
        if not self.depth_available:
            return None

        try:
            import torch

            # 转换图像
            input_batch = self.transform(frame).to(DEVICE)

            # 推理
            with torch.no_grad():
                prediction = self.model(input_batch)

                # 调整大小到原始分辨率
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # 转换为NumPy数组
            depth = prediction.cpu().numpy()

            # 归一化深度
            depth_min = depth.min()
            depth_max = depth.max()

            if depth_max - depth_min > 0:
                normalized_depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                normalized_depth = depth

            return normalized_depth

        except Exception as e:
            print(f"深度估计失败: {e}")
            return None


# ===== 跟踪器接口 =====
class TrackerInterface(ABC):
    """跟踪器接口"""

    @abstractmethod
    def init(self, frame, bbox):
        """初始化跟踪器

        Args:
            frame: 初始帧
            bbox: 边界框 (x, y, w, h)

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    def update(self, frame):
        """更新目标位置

        Args:
            frame: 当前帧

        Returns:
            tuple: (bool, bbox)，是否成功和边界框
        """
        pass


# ===== OpenCV传统跟踪器 =====
class OpenCVTracker(TrackerInterface):
    """使用OpenCV内置跟踪器"""

    def __init__(self, tracker_type='CSRT'):
        """初始化OpenCV跟踪器

        Args:
            tracker_type (str): 跟踪器类型，可选值包括:
                'CSRT', 'KCF', 'MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD'
        """
        self.tracker_type = tracker_type
        self.tracker = None

    def init(self, frame, bbox):
        """初始化跟踪器

        Args:
            frame: 初始帧
            bbox: 边界框 (x, y, w, h)

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 根据跟踪器类型创建跟踪器实例
            if self.tracker_type == 'CSRT':
                self.tracker = cv2.TrackerCSRT_create()
            elif self.tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            elif self.tracker_type == 'MOSSE':
                self.tracker = cv2.legacy.TrackerMOSSE_create()
            elif self.tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            elif self.tracker_type == 'BOOSTING':
                self.tracker = cv2.legacy.TrackerBoosting_create()
            elif self.tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.legacy.TrackerMedianFlow_create()
            elif self.tracker_type == 'TLD':
                self.tracker = cv2.legacy.TrackerTLD_create()
            else:
                print(f"不支持的跟踪器类型: {self.tracker_type}，使用默认的CSRT")
                self.tracker = cv2.TrackerCSRT_create()

            # 初始化跟踪器
            return self.tracker.init(frame, bbox)

        except Exception as e:
            print(f"OpenCV跟踪器初始化失败: {e}")
            return False

    def update(self, frame):
        """更新目标位置

        Args:
            frame: 当前帧

        Returns:
            tuple: (bool, bbox)，是否成功和边界框
        """
        if self.tracker is None:
            return False, None

        try:
            success, bbox = self.tracker.update(frame)
            return success, bbox
        except Exception as e:
            print(f"OpenCV跟踪器更新失败: {e}")
            return False, None


class DeepSORTTracker(TrackerInterface):
    def __init__(self, config):
        self.config = config
        self.initialized = False
        self.target_id = None
        self.init_bbox = None
        self.deepsort_available = False

        # 尝试导入DeepSORT相关模块
        try:
            # 导入DeepSort
            from deep_sort_realtime.deepsort_tracker import DeepSort

            # 注意：使用正确的参数名称，根据源码是embedder_gpu而不是use_cuda
            self.tracker = DeepSort(
                max_age=config.get('max_age', 30),
                n_init=config.get('n_init', 3),
                nms_max_overlap=config.get('nms_max_overlap', 1.0),
                max_cosine_distance=config.get('max_cosine_distance', 0.3),
                nn_budget=config.get('nn_budget', 100),
                embedder='mobilenet',  # 使用默认的mobilenet嵌入器
                embedder_gpu=TORCH_GPU_AVAILABLE,  # 正确的GPU参数
                half=True,  # 使用半精度
                bgr=True  # OpenCV使用BGR格式
            )

            self.deepsort_available = True
            print("DeepSORT跟踪器初始化成功")

        except ImportError:
            print("缺少DeepSORT所需库，将使用OpenCV跟踪器作为备选")
            print("安装命令: pip install deep-sort-realtime")
            self.tracker = OpenCVTracker()
            self.deepsort_available = False

        except Exception as e:
            print(f"DeepSORT初始化失败: {e}")
            print("将使用OpenCV跟踪器作为备选")
            self.tracker = OpenCVTracker()
            self.deepsort_available = False

    def init(self, frame, bbox):
        """初始化跟踪器

        Args:
            frame: 初始帧
            bbox: 边界框 (x, y, w, h)

        Returns:
            bool: 初始化是否成功
        """
        # 保存初始边界框
        self.init_bbox = bbox
        self.initialized = True

        # 如果不是DeepSORT，则使用OpenCV跟踪器
        if not self.deepsort_available:
            return self.tracker.init(frame, bbox)

        # DeepSORT不需要单独初始化，但我们需要记录初始状态
        self.last_frame = frame

        # 裁剪目标区域以进行第一次更新
        x, y, w, h = [int(v) for v in bbox]
        self.crop_area = (max(0, x - 10), max(0, y - 10),
                          min(frame.shape[1], x + w + 10), min(frame.shape[0], y + h + 10))

        return True

    def update(self, frame):
        if not self.initialized:
            return False, None

        if not self.deepsort_available:
            return self.tracker.update(frame)

        try:
            # 准备检测结果
            success, bbox = self.tracker_fallback.update(frame) if hasattr(self, 'tracker_fallback') else (True,
                                                                                                           self.init_bbox)
            if not success:
                return False, None

            # 将bbox格式转为[x, y, w, h]
            x, y, w, h = [int(v) for v in bbox]

            # Deep SORT需要的格式是[x, y, w, h, confidence, class_id]
            detection = [[x, y, w, h], 0.9, 0]  # 0.9是置信度，0是类别ID

            # 调用update_tracks，让DeepSORT内部处理嵌入提取
            tracks = self.tracker.update_tracks([detection], frame=frame)

            # 没有跟踪目标
            if not tracks:
                return False, None

            # 找到对应ID的跟踪目标
            if self.target_id is None and tracks:
                self.target_id = tracks[0].track_id

            for track in tracks:
                if track.track_id == self.target_id and track.is_confirmed():
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    return True, (x1, y1, x2 - x1, y2 - y1)

            return False, None

        except Exception as e:
            print(f"DeepSORT更新失败: {e}")
            # 创建一个备用跟踪器
            if not hasattr(self, 'tracker_fallback'):
                self.tracker_fallback = OpenCVTracker('CSRT')
                self.tracker_fallback.init(frame, self.init_bbox)
            return self.tracker_fallback.update(frame)


# ===== 高效轨迹数据结构 =====
class EfficientTrajectory:
    """高效的轨迹点存储和处理结构"""

    def __init__(self, max_points=1000):
        """初始化轨迹存储

        Args:
            max_points (int): 存储的最大轨迹点数量
        """
        # 固定大小的环形缓冲区，用于存储最近的轨迹点
        self.points = deque(maxlen=max_points)

        # 用于批量操作的NumPy数组
        self._np_array = None
        self._needs_update = False

    def append(self, point):
        """添加轨迹点

        Args:
            point: 要添加的轨迹点
        """
        self.points.append(point)
        self._needs_update = True

    def get_numpy_array(self):
        """获取轨迹点的NumPy数组表示，用于批量处理

        Returns:
            ndarray: 轨迹点的NumPy数组
        """
        if self._needs_update or self._np_array is None:
            self._np_array = np.array(list(self.points))
            self._needs_update = False
        return self._np_array

    def __len__(self):
        """获取轨迹点数量

        Returns:
            int: 轨迹点数量
        """
        return len(self.points)

    def __getitem__(self, idx):
        """获取指定索引的轨迹点

        Args:
            idx: 索引或切片

        Returns:
            轨迹点或轨迹点列表
        """
        if isinstance(idx, slice):
            return list(self.points)[idx]
        return list(self.points)[idx]


# ===== GPU加速工具 =====
class GPUAcceleratedTransforms:
    """提供GPU加速的点坐标变换"""

    def __init__(self):
        """初始化GPU加速工具"""
        # 检查CUDA是否可用
        self.cuda_available = GPU_AVAILABLE
        if not self.cuda_available:
            print("CUDA不可用，将使用CPU处理")

    def transform_points_gpu(self, points, matrix):
        """使用GPU加速变换点坐标

        Args:
            points (list): 要变换的点列表
            matrix (ndarray): 变换矩阵

        Returns:
            list: 变换后的点列表
        """
        if not self.cuda_available or len(points) < 500:
            # 对于小批量的点，CPU可能比GPU更快，因为GPU传输开销
            return self._transform_points_cpu(points, matrix)

        try:
            # 转换为适合GPU处理的格式
            points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

            # 上传到GPU
            gpu_points = cv2.cuda_GpuMat()
            gpu_points.upload(points_array)

            # GPU上进行变换
            gpu_result = cv2.cuda.perspectiveTransform(gpu_points, matrix)

            # 下载结果
            result = gpu_result.download()
            return result.reshape(-1, 2).tolist()

        except Exception as e:
            print(f"GPU变换失败: {e}，回退到CPU")
            return self._transform_points_cpu(points, matrix)

    def _transform_points_cpu(self, points, matrix):
        """CPU备用点坐标变换

        Args:
            points (list): 要变换的点列表
            matrix (ndarray): 变换矩阵

        Returns:
            list: 变换后的点列表
        """
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_array, matrix)
        return transformed.reshape(-1, 2).tolist()


# ===== 运动估计器接口 =====
class MotionEstimatorInterface(ABC):
    """相机运动估计器接口"""

    @abstractmethod
    def estimate(self, prev_frame, curr_frame, mask=None):
        """估计两帧之间的相机运动

        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧
            mask: 掩码，指定不参与运动估计的区域

        Returns:
            ndarray: 变换矩阵
        """
        pass


# ===== 光流运动估计器 =====
class OpticalFlowMotionEstimator(MotionEstimatorInterface):
    """基于光流的相机运动估计"""

    def __init__(self, config):
        """初始化光流运动估计器

        Args:
            config (dict): 配置参数
        """
        self.config = config
        self.max_corners = config.get('max_corners', 100)
        self.quality_level = config.get('quality_level', 0.01)
        self.min_distance = config.get('min_distance', 15)
        self.block_size = config.get('block_size', 3)
        self.visualize = config.get('visualize_features', False)

    def estimate(self, prev_frame, curr_frame, mask=None):
        """估计两帧之间的相机运动

        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧
            mask: 掩码，指定不参与运动估计的区域

        Returns:
            ndarray: 变换矩阵
        """
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 使用goodFeaturesToTrack检测角点
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=mask
        )

        # 创建可视化帧
        if self.visualize:
            vis_frame = curr_frame.copy()

        # 如果没有找到特征点，返回单位矩阵
        if prev_pts is None or len(prev_pts) < 4:
            return np.eye(3, dtype=np.float32)

        # 可视化初始特征点
        if self.visualize:
            for pt in prev_pts:
                x, y = pt[0]
                cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # 使用光流跟踪特征点
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None
        )

        # 过滤有效的特征点
        idx = np.where(status == 1)[0]
        if len(idx) < 4:
            return np.eye(3, dtype=np.float32)

        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # 可视化跟踪的特征点
        if self.visualize:
            for i, (prev, curr) in enumerate(zip(prev_pts, curr_pts)):
                px, py = prev[0]
                cx, cy = curr[0]
                cv2.circle(vis_frame, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                cv2.line(vis_frame, (int(px), int(py)), (int(cx), int(cy)), (255, 0, 0), 1)

        # 使用RANSAC估计仿射变换
        m, inliers = cv2.estimateAffinePartial2D(
            prev_pts, curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99
        )

        # 可视化RANSAC内点
        if self.visualize and inliers is not None:
            inlier_count = np.sum(inliers)
            outlier_count = len(inliers) - inlier_count

            for i, is_inlier in enumerate(inliers):
                if is_inlier:
                    cx, cy = curr_pts[i][0]
                    cv2.circle(vis_frame, (int(cx), int(cy)), 5, (0, 0, 255), 1)

            # 显示统计信息
            info_text = f"Features: {len(prev_pts)}, Inliers: {inlier_count}, Outliers: {outlier_count}"
            cv2.putText(vis_frame, str(info_text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # 显示可视化
            cv2.imshow('Feature Tracking', vis_frame)

        # 如果估计失败，返回单位矩阵
        if m is None:
            return np.eye(3, dtype=np.float32)

        # 转换为齐次变换矩阵
        m_homo = np.vstack([m, [0, 0, 1]])

        return m_homo


# ===== 深度辅助的运动估计器 =====
class DepthAwareMotionEstimator(MotionEstimatorInterface):
    """基于深度信息的相机运动估计"""

    def __init__(self, config):
        """初始化深度辅助的运动估计器

        Args:
            config (dict): 配置参数
        """
        self.config = config
        self.depth_estimator = DepthEstimator(config)

        # 光流参数
        self.max_corners = config.get('max_corners', 100)
        self.quality_level = config.get('quality_level', 0.01)
        self.min_distance = config.get('min_distance', 15)
        self.block_size = config.get('block_size', 3)
        self.visualize = config.get('visualize_features', False)

        # 备用光流估计器
        self.optical_flow_estimator = OpticalFlowMotionEstimator(config)

    def estimate(self, prev_frame, curr_frame, mask=None):
        """估计两帧之间的相机运动，使用深度信息

        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧
            mask: 掩码，指定不参与运动估计的区域

        Returns:
            ndarray: 变换矩阵
        """
        # 检查深度估计器是否可用
        if not self.depth_estimator.depth_available:
            # 如果深度不可用，回退到光流
            return self.optical_flow_estimator.estimate(prev_frame, curr_frame, mask)

        try:
            # 估计深度图
            prev_depth = self.depth_estimator.estimate_depth(prev_frame)
            curr_depth = self.depth_estimator.estimate_depth(curr_frame)

            # 检查深度估计是否成功
            if prev_depth is None or curr_depth is None:
                return self.optical_flow_estimator.estimate(prev_frame, curr_frame, mask)

            # 转换为灰度图
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # 使用goodFeaturesToTrack检测角点
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                mask=mask
            )

            # 如果没有找到足够的特征点，回退到光流
            if prev_pts is None or len(prev_pts) < 8:
                return self.optical_flow_estimator.estimate(prev_frame, curr_frame, mask)

            # 使用光流跟踪特征点
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None
            )

            # 过滤有效的特征点
            idx = np.where(status == 1)[0]
            if len(idx) < 8:
                return self.optical_flow_estimator.estimate(prev_frame, curr_frame, mask)

            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # 收集特征点的深度
            prev_depths = []
            curr_depths = []

            for i, (prev, curr) in enumerate(zip(prev_pts, curr_pts)):
                px, py = prev[0]
                cx, cy = curr[0]

                # 确保坐标在图像范围内
                px, py = int(min(max(px, 0), prev_frame.shape[1] - 1)), int(min(max(py, 0), prev_frame.shape[0] - 1))
                cx, cy = int(min(max(cx, 0), curr_frame.shape[1] - 1)), int(min(max(cy, 0), curr_frame.shape[0] - 1))

                # 获取深度
                pd = prev_depth[py, px]
                cd = curr_depth[cy, cx]

                prev_depths.append(pd)
                curr_depths.append(cd)

            prev_depths = np.array(prev_depths)
            curr_depths = np.array(curr_depths)

            # 判断深度比例变化是否过大
            depth_ratio = np.median(prev_depths / np.clip(curr_depths, 1e-5, None))
            if depth_ratio < 0.5 or depth_ratio > 2.0:
                # 深度变化太大，可能不可靠，回退到光流
                return self.optical_flow_estimator.estimate(prev_frame, curr_frame, mask)

            # 可视化
            if self.visualize:
                vis_frame = curr_frame.copy()

                # 绘制特征点
                for i, (prev, curr) in enumerate(zip(prev_pts, curr_pts)):
                    px, py = prev[0]
                    cx, cy = curr[0]

                    # 颜色编码深度（红色=近，蓝色=远）
                    depth_val = curr_depths[i]
                    color = (int(255 * (1 - depth_val)), 0, int(255 * depth_val))

                    cv2.circle(vis_frame, (int(cx), int(cy)), 3, color, -1)
                    cv2.line(vis_frame, (int(px), int(py)), (int(cx), int(cy)), color, 1)

                # 显示深度统计信息
                cv2.putText(vis_frame, f"Depth Ratio: {depth_ratio:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 显示深度图
                depth_vis = (curr_depth * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # 调整大小适合屏幕
                scale = min(1.0, 800 / depth_color.shape[1])
                depth_color = cv2.resize(depth_color, None, fx=scale, fy=scale)

                cv2.imshow('Depth Map', depth_color)
                cv2.imshow('Feature Tracking (Depth)', vis_frame)

            # 用深度加权的RANSAC估计仿射变换
            weights = 1.0 / np.clip(curr_depths, 0.1, 1.0)  # 给近处的点更高的权重

            # 使用RANSAC估计仿射变换
            m, inliers = cv2.estimateAffinePartial2D(
                prev_pts, curr_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )

            # 如果估计失败，回退到光流
            if m is None:
                return self.optical_flow_estimator.estimate(prev_frame, curr_frame, mask)

            # 转换为齐次变换矩阵
            m_homo = np.vstack([m, [0, 0, 1]])

            return m_homo

        except Exception as e:
            print(f"深度辅助运动估计失败: {e}")
            # 出错时回退到光流
            return self.optical_flow_estimator.estimate(prev_frame, curr_frame, mask)


class ExplorationMapManager:
    """全局探索地图管理器 - 像素级视野探索追踪"""

    def __init__(self, map_size, initial_transform):
        """初始化探索地图管理器

        Args:
            map_size (tuple): 全局地图尺寸 (width, height)
            initial_transform (ndarray): 初始坐标变换矩阵
        """
        self.map_width, self.map_height = map_size

        # 全局探索地图 - 二值图像，255表示已探索，0表示未探索
        self.exploration_map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        # 存储所有标注的全局像素掩码
        self.annotation_masks = {
            'polygons': [],  # [(mask, bbox), ...]
            'circles': []  # [(mask, bbox), ...]
        }

        # 用于优化的边界框缓存
        self.dirty_regions = []  # 需要更新的区域

    def create_sector_mask(self, center, radius, facing_angle, sector_angle, map_size):
        """创建扇形区域的掩码

        Args:
            center (tuple): 扇形中心点 (x, y)
            facing_angle (float): 朝向角度（弧度）
            sector_angle (float): 扇形角度（弧度）
            radius (float): 扇形半径
            map_size (tuple): 地图尺寸 (width, height)

        Returns:
            ndarray: 扇形掩码，255表示扇形内，0表示扇形外
        """
        mask = np.zeros((map_size[1], map_size[0]), dtype=np.uint8)

        # 计算扇形的起始和结束角度
        half_angle = sector_angle / 2
        start_angle = facing_angle - half_angle
        end_angle = facing_angle + half_angle

        # 创建坐标网格
        y_coords, x_coords = np.ogrid[:map_size[1], :map_size[0]]

        # 计算每个像素到中心的距离和角度
        dx = x_coords - center[0]
        dy = y_coords - center[1]
        distances = np.sqrt(dx ** 2 + dy ** 2)
        angles = np.arctan2(dy, dx)

        # 处理角度跨越-π到π边界的情况
        if end_angle - start_angle > np.pi:
            # 扇形跨越-π到π边界
            angle_mask = (angles >= start_angle) | (angles <= end_angle - 2 * np.pi)
        elif start_angle < -np.pi:
            # 起始角度小于-π
            start_angle += 2 * np.pi
            end_angle += 2 * np.pi
            angles = np.where(angles < 0, angles + 2 * np.pi, angles)
            angle_mask = (angles >= start_angle) & (angles <= end_angle)
        elif end_angle > np.pi:
            # 结束角度大于π
            start_angle -= 2 * np.pi
            end_angle -= 2 * np.pi
            angles = np.where(angles > 0, angles - 2 * np.pi, angles)
            angle_mask = (angles >= start_angle) & (angles <= end_angle)
        else:
            # 正常情况
            angle_mask = (angles >= start_angle) & (angles <= end_angle)

        # 组合距离和角度条件
        sector_mask = (distances <= radius) & angle_mask
        mask[sector_mask] = 255

        return mask

    def update_exploration(self, robot_position, robot_facing, sector_radius, sector_angle):
        """更新探索地图

        Args:
            robot_position (tuple): 机器人在全局坐标系中的位置
            robot_facing (float): 机器人朝向角度（弧度）
            sector_radius (float): 扇形半径
            sector_angle (float): 扇形角度（弧度）
        """
        # 创建当前帧的扇形掩码
        sector_mask = self.create_sector_mask(
            robot_position, sector_radius, robot_facing, sector_angle,
            (self.map_width, self.map_height)
        )

        # 更新探索地图（使用或运算，已探索的区域保持探索状态）
        self.exploration_map = cv2.bitwise_or(self.exploration_map, sector_mask)

    def add_annotation_mask(self, annotation_type, points_or_center, radius=None):
        """添加标注的全局掩码

        Args:
            annotation_type (str): 'polygon' 或 'circle'
            points_or_center: 多边形点列表或圆心坐标
            radius (float): 圆半径（仅圆形需要）

        Returns:
            int: 标注索引
        """
        mask = np.zeros((self.map_height, self.map_width), dtype=np.uint8)

        if annotation_type == 'polygon':
            # 创建多边形掩码
            points = np.array(points_or_center, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

            # 计算边界框
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            bbox = (np.min(x_coords), np.min(y_coords),
                    np.max(x_coords), np.max(y_coords))

            self.annotation_masks['polygons'].append((mask, bbox))
            return len(self.annotation_masks['polygons']) - 1

        elif annotation_type == 'circle':
            # 创建圆形掩码
            center = (int(points_or_center[0]), int(points_or_center[1]))
            cv2.circle(mask, center, int(radius), 255, -1)

            # 计算边界框
            bbox = (center[0] - radius, center[1] - radius,
                    center[0] + radius, center[1] + radius)

            self.annotation_masks['circles'].append((mask, bbox))
            return len(self.annotation_masks['circles']) - 1

    def get_explored_annotation_pixels(self, annotation_type, annotation_idx):
        """获取已探索的标注像素掩码

        Args:
            annotation_type (str): 'polygon' 或 'circle'
            annotation_idx (int): 标注索引

        Returns:
            ndarray: 已探索的标注像素掩码
        """
        if annotation_type == 'polygon':
            if annotation_idx >= len(self.annotation_masks['polygons']):
                return None
            annotation_mask, bbox = self.annotation_masks['polygons'][annotation_idx]
        elif annotation_type == 'circle':
            if annotation_idx >= len(self.annotation_masks['circles']):
                return None
            annotation_mask, bbox = self.annotation_masks['circles'][annotation_idx]
        else:
            return None

        # 计算标注掩码与探索地图的交集
        explored_pixels = cv2.bitwise_and(annotation_mask, self.exploration_map)

        return explored_pixels

    def clear_annotations(self):
        """清除所有标注掩码"""
        self.annotation_masks['polygons'].clear()
        self.annotation_masks['circles'].clear()

    def get_exploration_coverage(self):
        """获取探索覆盖率

        Returns:
            float: 探索覆盖率 (0-1)
        """
        total_pixels = self.map_width * self.map_height
        explored_pixels = np.sum(self.exploration_map > 0)
        return explored_pixels / total_pixels if total_pixels > 0 else 0.0

# ===== 视频目标跟踪与标注工具 =====
class VideoTracker:
    """视频目标追踪与标注工具主类"""

    def __init__(self, config):
        """初始化视频追踪器

        Args:
            config (dict): 配置参数字典
        """
        # 保存配置参数
        self.config = config

        # 初始状态设为初始化中
        self.current_state = State.INITIALIZING
        self.current_frame = None
        self.display_frame = None

        # 初始化窗口名称和回调标志
        self.main_window_name = 'Trajectory'
        self.callbacks_registered = False

        # 性能监控
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()

        # 初始化变量
        self.feature_vis_data = None
        self.fitted_rectangle = None
        self.orientation_dragging = False

        # 创建特征提取器
        if self.config.get('use_cnn_features', False):
            self.feature_extractor = CNNFeatureExtractor(config)
            print("使用CNN特征提取器")
        else:
            self.feature_extractor = SIFTFeatureExtractor(config)
            print("使用SIFT特征提取器")

        # 创建运动估计器
        if self.config.get('use_depth_motion', False):
            self.motion_estimator = DepthAwareMotionEstimator(config)
            print("使用深度辅助的运动估计器")
        else:
            self.motion_estimator = OpticalFlowMotionEstimator(config)
            print("使用光流运动估计器")

        # 初始化追踪相关变量
        self.trajectory = None
        self.prev_position = None
        self.direction_angles = None
        self.trajectory_with_direction = None
        self.kalman = None
        self.has_angle = None
        self.template_features = None
        self.template_roi = None
        self.initial_orientation = None
        self.orientation_setup_data = None

        # 标注相关变量
        self.current_polygon_points = []
        self.polygons = []
        self.circles = []
        self.circle_start_point = None
        self.current_circle_radius = None
        self.temp_circle = None
        self.is_drawing_circle = False

        # 初始化视频捕获器
        self.setup_video_capture()

        # 如果需要水印，创建水印
        self.watermark = None
        self.watermark_mask = None
        if self.config['add_watermark']:
            self.create_watermark()

        # 开始初始化工作流程
        self.initialize_tracking_workflow()
        self.prev_frame = self.first_frame.copy()  # 初始化前一帧彩色图像

        self.frame_data = []  # 记录每帧的完整信息
        self.camera_transforms = []  # 记录相机运动轨迹
        self.bounding_boxes = []  # 记录每帧的追踪框

    def export_tracking_data(self, export_path="tracking_data_5_29_good.json"):
        """导出追踪数据供refine阶段使用 - 彻底修复JSON序列化问题"""

        # 验证数据完整性
        if len(self.trajectory) == 0:
            print("没有轨迹数据可导出")
            return

        # 计算轨迹边界框用于后续地图尺寸计算 - 确保类型转换
        trajectory_array = self.trajectory.get_numpy_array()
        trajectory_positions = [[float(point[0]), float(point[1])] for point in trajectory_array]

        # 计算轨迹边界
        positions_array = np.array(trajectory_positions)
        min_coords = positions_array.min(axis=0)
        max_coords = positions_array.max(axis=0)

        # 计算相机轨迹
        camera_positions, camera_orientations = self.calculate_camera_trajectory()

        # 计算相机轨迹边界（用于确定全局地图范围）
        if len(camera_positions) > 1:
            camera_array = np.array(camera_positions)
            camera_min_coords = camera_array.min(axis=0)
            camera_max_coords = camera_array.max(axis=0)

            # 合并机器人和相机的边界来确定整体探索范围
            combined_min = np.minimum(min_coords, camera_min_coords)
            combined_max = np.maximum(max_coords, camera_max_coords)
        else:
            combined_min = min_coords
            combined_max = max_coords

        # 确保数据长度一致性
        min_length = min(len(trajectory_positions), len(self.direction_angles),
                         len(self.camera_transforms), len(self.bounding_boxes))

        if min_length != len(trajectory_positions):
            print(f"警告：数据长度不一致，截取到最短长度 {min_length}")
            trajectory_positions = trajectory_positions[:min_length]
            self.direction_angles = self.direction_angles[:min_length]
            self.camera_transforms = self.camera_transforms[:min_length]
            self.bounding_boxes = self.bounding_boxes[:min_length]
            self.frame_data = self.frame_data[:min_length]

        # 构建原始数据结构
        raw_tracking_data = {
            'video_info': {
                'path': self.config['video_path'],
                'output_path': self.config.get('output_video_path', 'refined_output.mp4'),
                'fps': self.fps,
                'width': self.frame_width,
                'height': self.frame_height,
                'total_processed_frames': min_length
            },

            'trajectory_summary': {
                'total_points': len(trajectory_positions),
                'bounds': {
                    'min_x': min_coords[0],
                    'min_y': min_coords[1],
                    'max_x': max_coords[0],
                    'max_y': max_coords[1]
                },
                'trajectory_positions': trajectory_positions,
                'direction_angles': self.direction_angles[:min_length]
            },

            # 相机运动信息
            'camera_motion': {
                'camera_positions': camera_positions[:min_length] if len(
                    camera_positions) > min_length else camera_positions,
                'camera_orientations': camera_orientations[:min_length] if len(
                    camera_orientations) > min_length else camera_orientations,
                'camera_bounds': {
                    'min_x': combined_min[0],
                    'min_y': combined_min[1],
                    'max_x': combined_max[0],
                    'max_y': combined_max[1]
                },
                'transform_matrices': [transform.tolist() for transform in self.camera_transforms[:min_length]]
            },

            # 每帧详细数据
            'frame_data': self.frame_data[:min_length],

            # bounding box信息
            'tracking_boxes': self.bounding_boxes[:min_length],

            'annotations': {
                'polygons': [
                    {
                        'points': [[p[0], p[1]] for p in poly],
                        'global_coords': True
                    }
                    for poly in self.polygons
                ],
                'circles': [
                    {
                        'center': [center[0], center[1]],
                        'radius': radius,
                        'global_coords': True
                    }
                    for center, radius in self.circles
                ]
            },

            'render_config': {
                'sector_radius': self.config['sector_radius'],
                'sector_angle': self.config['sector_angle'],
                'trajectory_color': self.config['trajectory_color'],
                'trajectory_thickness': self.config['trajectory_thickness'],
                'trajectory_alpha': self.config['trajectory_alpha'],
                'direction_arrow_length': self.config['direction_arrow_length'],
                'direction_arrow_color': self.config['direction_arrow_color'],
                'arrow_alpha': self.config['arrow_alpha'],
                'sector_color': self.config['sector_color'],
                'sector_alpha': self.config['sector_alpha'],
                'polygon_fill_color': self.config['polygon_fill_color'],
                'polygon_border_color': self.config['polygon_border_color'],
                'polygon_border_thickness': self.config['polygon_border_thickness'],
                'polygon_alpha': self.config['polygon_alpha'],
                'circle_fill_color': self.config['circle_fill_color'],
                'circle_border_color': self.config['circle_border_color'],
                'circle_border_thickness': self.config['circle_border_thickness'],
                'circle_alpha': self.config['circle_alpha'],

                # bounding box渲染配置
                'bbox_color': (255, 0, 0),
                'bbox_thickness': 2,
                'show_bbox': True,
                'show_fps': self.config.get('show_fps', False),
                'show_fitted_rectangle': self.config.get('show_fitted_rectangle', True)
            },

            # 添加数据验证信息
            'data_validation': {
                'export_timestamp': time.time(),
                'data_lengths': {
                    'trajectory': len(trajectory_positions),
                    'angles': len(self.direction_angles[:min_length]),
                    'transforms': len(self.camera_transforms[:min_length]),
                    'bboxes': len(self.bounding_boxes[:min_length]),
                    'frame_data': len(self.frame_data[:min_length])
                },
                'bounds_info': {
                    'trajectory_bounds': {
                        'min_x': min_coords[0], 'min_y': min_coords[1],
                        'max_x': max_coords[0], 'max_y': max_coords[1]
                    },
                    'combined_bounds': {
                        'min_x': combined_min[0], 'min_y': combined_min[1],
                        'max_x': combined_max[0], 'max_y': combined_max[1]
                    }
                }
            }
        }

        # 对整个数据结构进行深度NumPy类型转换
        print("正在转换数据类型...")
        tracking_data = self.convert_numpy_types(raw_tracking_data)

        # 保存数据
        import json
        try:
            print("开始保存JSON文件...")
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2, ensure_ascii=False)

            print(f"追踪数据已导出到: {export_path}")
            print(f"数据统计:")
            print(f"  轨迹点数: {len(trajectory_positions)}")
            print(f"  相机轨迹点数: {len(camera_positions)}")
            print(f"  帧数据记录数: {min_length}")
            print(f"  追踪框记录数: {len(self.bounding_boxes[:min_length])}")
            print(f"  多边形数: {len(self.polygons)}")
            print(f"  圆形数: {len(self.circles)}")
            print(
                f"  机器人轨迹边界: X({float(min_coords[0]):.1f}, {float(max_coords[0]):.1f}), Y({float(min_coords[1]):.1f}, {float(max_coords[1]):.1f})")
            print(
                f"  整体探索边界: X({float(combined_min[0]):.1f}, {float(combined_max[0]):.1f}), Y({float(combined_min[1]):.1f}, {float(combined_max[1]):.1f})")

        except Exception as e:
            print(f"导出数据失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    def setup_video_capture(self):
        """设置视频捕获和输出"""
        try:
            # 打开视频文件
            self.cap = cv2.VideoCapture(self.config['video_path'])

            # 检查视频是否成功打开
            if not self.cap.isOpened():
                print("无法打开视频文件")
                exit()

            # 获取视频帧率和大小
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 如果需要记录视频，初始化视频写入器
            if self.config['record_video']:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(
                    self.config['output_video_path'],
                    fourcc,
                    self.fps,
                    (self.frame_width, self.frame_height)
                )

            # 读取第一帧
            ret, first_frame = self.cap.read()
            if not ret:
                print("无法读取视频帧")
                exit()

            self.first_frame = first_frame
            self.prev_gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)

            # 初始化累计变换矩阵为单位矩阵
            self.cumulative_transform = np.eye(3, 3, dtype=np.float32)

        except Exception as e:
            print(f"视频设置错误: {e}")
            exit()

    def initialize_tracking_workflow(self):
        """初始化跟踪工作流程

        重新设计初始化流程，确保窗口和回调以正确的顺序创建
        """
        print("\n=== 开始初始化跟踪流程 ===")
        print("步骤1: 请选择要追踪的目标")

        # 第一步：选择追踪目标
        self.tracking_bbox = self.select_tracking_target()

        # 第二步：初始化其他变量
        self.initialize_tracking_variables()
        self.initialize_annotation_variables()

        # 第三步：设置朝向模板（可选）
        if self.config.get('skip_initial_orientation_setup', False) and len(self.template_library) > 0:
            print("\n已加载保存的模板，跳过初始朝向设置步骤...")
            # 从首个模板获取初始朝向
            self.initial_orientation = self.template_library[0]['orientation']
            self.last_template_orientation = self.initial_orientation
            print(f"使用已加载模板的朝向作为初始朝向: {np.degrees(self.initial_orientation):.1f}度")
        else:
            # 正常设置朝向模板
            print("\n步骤2: 请设置目标初始朝向")
            self.setup_orientation_template()

        # 第四步：创建主窗口并设置鼠标回调
        print("\n初始化完成！开始追踪...")
        cv2.namedWindow(self.main_window_name)
        cv2.setMouseCallback(self.main_window_name, self.mouse_callback)
        self.callbacks_registered = True

        # 创建调试窗口（如果需要）
        if self.config.get('debug_orientation', False):
            cv2.namedWindow('Orientation Debug', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Orientation Debug', 800, 400)

        if self.config.get('debug_feature_matching', False):
            cv2.namedWindow('Feature Matching', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Feature Matching', 800, 400)

        # 更改状态为播放
        self.current_state = State.PLAYING

    def select_tracking_target(self):
        """让用户选择要追踪的目标

        Returns:
            tuple: 选中的边界框 (x, y, w, h)
        """
        # 创建选择窗口
        select_window_name = 'Select Target'
        cv2.namedWindow(select_window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(select_window_name, self.first_frame)

        print("请使用鼠标选择要追踪的目标，然后按回车确认")
        bbox = cv2.selectROI(select_window_name, self.first_frame, False)

        # 关闭选择窗口
        cv2.destroyWindow(select_window_name)

        # 初始化追踪器
        try:
            # 根据配置选择跟踪器
            tracker_type = self.config.get('tracker_type', 'CSRT')

            if tracker_type.upper() == 'DEEPSORT':
                self.tracker = DeepSORTTracker(self.config)
                print("使用DeepSORT跟踪器")
            else:
                self.tracker = OpenCVTracker(tracker_type)
                print(f"使用OpenCV {tracker_type}跟踪器")

            # 初始化跟踪器
            success = self.tracker.init(self.first_frame, bbox)

            if not success:
                print("跟踪器初始化失败，使用备选跟踪器")
                self.tracker = OpenCVTracker('CSRT')
                self.tracker.init(self.first_frame, bbox)

            print(f"目标选择成功: {bbox}")
            return bbox
        except Exception as e:
            print(f"追踪器初始化失败: {e}")
            print("使用默认CSRT跟踪器")
            self.tracker = OpenCVTracker('CSRT')
            self.tracker.init(self.first_frame, bbox)
            return bbox

    def setup_orientation_template(self):
        """设置目标朝向模板和初始朝向，使用键盘控制角度旋转"""
        # 获取边界框
        x, y, w, h = self.tracking_bbox

        # 提取目标区域作为模板
        expansion = self.config.get('template_expansion', 10)
        ex_x = max(0, x - expansion // 2)
        ex_y = max(0, y - expansion // 2)
        ex_w = min(self.first_frame.shape[1] - ex_x, w + expansion)
        ex_h = min(self.first_frame.shape[0] - ex_y, h + expansion)

        # 保存模板ROI
        self.template_roi = self.first_frame[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w].copy()

        # 创建朝向设置窗口
        orientation_window = "设置初始朝向"
        cv2.namedWindow(orientation_window)

        # 初始化变量
        center_x, center_y = ex_w // 2, ex_h // 2
        current_angle = -90  # 默认朝上，角度制
        angle_step = 5  # 每次旋转的角度，减少为5度提高精度
        arrow_length = 30

        # 根据当前角度计算箭头终点
        angle_rad = np.radians(current_angle)
        direction_x = center_x + int(arrow_length * np.cos(angle_rad))
        direction_y = center_y + int(arrow_length * np.sin(angle_rad))

        # 初始化orientation_setup_data
        self.orientation_setup_data = {
            'center_x': center_x,
            'center_y': center_y,
            'direction_x': direction_x,
            'direction_y': direction_y,
            'template_with_arrow': self.template_roi.copy(),
            'window_name': orientation_window,
            'current_angle': current_angle
        }

        print("\n请使用键盘设置初始朝向:")
        print("←/A: 逆时针旋转箭头")
        print("→/D: 顺时针旋转箭头")
        print("↑: 增加旋转步长")
        print("↓: 减小旋转步长")
        print("Enter: 确认朝向")
        print("ESC: 重置为默认朝向")

        while True:
            # 创建显示图像
            display = self.template_roi.copy()

            # 使用线段替代箭头 (将cv2.arrowedLine改为cv2.line)
            cv2.line(display, (center_x, center_y), (direction_x, direction_y), (0, 255, 0), 2)

            # 其余文字提示和操作保持不变
            cv2.putText(display, "使用左右方向键或A/D键调整角度", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display, "完成后按回车确认", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 绘制当前角度和步长
            cv2.putText(display, f"当前角度: {current_angle:.1f}°", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(display, f"步长: {angle_step:.1f}°", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 更新orientation_setup_data中的值
            self.orientation_setup_data['direction_x'] = direction_x
            self.orientation_setup_data['direction_y'] = direction_y
            self.orientation_setup_data['current_angle'] = current_angle
            self.orientation_setup_data['template_with_arrow'] = display.copy()

            # 显示图像
            cv2.imshow(orientation_window, display)

            # 获取键盘输入
            key = cv2.waitKey(20) & 0xFF

            # 如果按下回车，确认朝向并退出
            if key == 13:  # 回车键
                break

            # 如果按下ESC，重置为默认朝向
            if key == 27:  # ESC键
                current_angle = -90  # 重置为朝上
                angle_rad = np.radians(current_angle)
                direction_x = center_x + int(arrow_length * np.cos(angle_rad))
                direction_y = center_y + int(arrow_length * np.sin(angle_rad))
                continue

            # 根据按键调整方向
            if key == ord('a') or key == 81:  # A 或 左箭头: 逆时针旋转
                current_angle -= angle_step
            elif key == ord('d') or key == 83:  # D 或 右箭头: 顺时针旋转
                current_angle += angle_step
            elif key == 82:  # 上箭头: 增加步长
                angle_step = min(angle_step + 1, 45)  # 最大步长45度
            elif key == 84:  # 下箭头: 减小步长
                angle_step = max(angle_step - 1, 1)  # 最小步长1度

            # 限制角度在[-180, 180]范围内
            current_angle = ((current_angle + 180) % 360) - 180

            # 更新箭头终点坐标
            angle_rad = np.radians(current_angle)
            direction_x = center_x + int(arrow_length * np.cos(angle_rad))
            direction_y = center_y + int(arrow_length * np.sin(angle_rad))

        # 计算初始朝向角度（弧度）
        self.initial_orientation = np.radians(current_angle)
        print(f"初始朝向角度: {current_angle:.1f}度")

        # 提取模板特征
        self.extract_template_features()
        self.generate_rotated_templates()
        # 保存初始模板到磁盘
        if self.config.get('template_save_path', None):
            try:
                self.save_template(self.template_roi, self.initial_orientation)
                print("已保存初始朝向模板")
            except Exception as e:
                print(f"保存初始朝向模板失败: {e}")

        # 关闭窗口
        cv2.destroyWindow(orientation_window)

    def orientation_mouse_callback(self, event, x, y, flags, param):
        """朝向设置过程中的鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始拖动，记录初始位置
            self.orientation_dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            # 更新箭头方向
            self.orientation_setup_data['direction_x'] = x
            self.orientation_setup_data['direction_y'] = y

            # 更新显示
            temp = self.template_roi.copy()
            cv2.arrowedLine(temp,
                            (self.orientation_setup_data['center_x'],
                             self.orientation_setup_data['center_y']),
                            (x, y),
                            (0, 255, 0), 2)

            # 添加说明文字
            cv2.putText(temp, "请拖动鼠标设置初始朝向", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(temp, "完成后按回车确认", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            self.orientation_setup_data['template_with_arrow'] = temp

        elif event == cv2.EVENT_LBUTTONUP:
            # 结束拖动
            self.orientation_dragging = False

    def extract_template_features(self):
        """从模板提取特征"""
        try:
            # 使用特征提取器提取模板特征
            self.template_features = self.feature_extractor.extract_features(self.template_roi)

            # 将初始模板添加到模板库
            self.template_library.append({
                'roi': self.template_roi.copy(),
                'features': self.template_features,
                'orientation': self.initial_orientation,
                'usage_count': 0,
                'quality': 1.0  # 初始模板假定质量最高
            })
            self.last_template_orientation = self.initial_orientation

            keypoints = self.template_features.get('keypoints', [])

            if len(keypoints) < 10:
                print(f"警告: 模板中仅检测到{len(keypoints)}个特征点，可能不足以可靠估计朝向")
            else:
                print(f"成功从模板中提取了{len(keypoints)}个特征点")

            # 在模板上绘制特征点（如果需要调试）
            if self.config.get('debug_feature_matching', False):
                template_vis = cv2.drawKeypoints(self.template_roi, keypoints, None,
                                                 color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('模板特征点', template_vis)
                cv2.waitKey(1000)  # 显示1秒
                cv2.destroyWindow('模板特征点')

        except Exception as e:
            print(f"提取模板特征失败: {e}")
            self.template_features = None

    def generate_rotated_templates(self):
        """生成360度旋转模板库"""
        if len(self.template_library) == 0:
            print("无初始模板可用于生成旋转模板")
            return

        print("正在生成旋转模板库...")
        # 使用第一个模板作为基础
        base_template = self.template_library[0]
        base_roi = base_template['roi']
        base_orientation = base_template['orientation']

        # 计算旋转中心
        h, w = base_roi.shape[:2]
        center = (w // 2, h // 2)

        # 每20度生成一个新模板
        for angle_deg in range(0, 360, 60):
            # 跳过与基础模板角度相近的
            base_angle_deg = np.degrees(base_orientation) % 360
            if abs((angle_deg - base_angle_deg) % 360) < 10:
                continue

            # 计算旋转角度（相对于基础模板的角度差）
            angle_diff = angle_deg - base_angle_deg

            # 生成旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle_diff, 1.0)
            # 执行仿射变换
            rotated_roi = cv2.warpAffine(base_roi, rotation_matrix, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)

            # 计算旋转后的朝向角度（弧度）
            new_orientation = (base_orientation + np.radians(angle_diff)) % (2 * np.pi)

            # 提取特征
            features = self.feature_extractor.extract_features(rotated_roi)

            # 添加到模板库
            self.template_library.append({
                'roi': rotated_roi,
                'features': features,
                'orientation': new_orientation,
                'usage_count': 0,
                'quality': 0.9,  # 合成模板质量稍低于原始模板
                'manual_calibrated': False,
                'synthetic': True  # 标记为合成模板
            })

        print(f"已生成 {len(self.template_library) - 1} 个额外的旋转模板")

    def find_best_template(self, current_angle):
        """查找最佳匹配的模板

        Args:
            current_angle: 当前预估朝向角度

        Returns:
            int: 最佳匹配模板的索引
        """
        if not self.template_library:
            return 0  # 如果模板库为空，返回默认模板

        # 计算当前角度与各模板角度的差异
        min_diff = float('inf')
        best_idx = 0

        for i, template in enumerate(self.template_library):
            # 计算角度差异（考虑循环性）
            angle_diff = abs(np.arctan2(np.sin(current_angle - template['orientation']),
                                        np.cos(current_angle - template['orientation'])))
            if angle_diff < min_diff:
                min_diff = angle_diff
                best_idx = i

        # 更新使用次数
        self.template_library[best_idx]['usage_count'] += 1
        return best_idx

    def should_add_new_template(self, current_angle, match_quality):
        """判断是否应该添加新模板

        Args:
            current_angle: 当前朝向角度
            match_quality: 匹配质量指标(0-1)

        Returns:
            bool: 是否应添加新模板
        """
        if not self.template_library:
            return True

        # 如果匹配质量低，且角度与现有模板差异大，添加新模板
        if match_quality < 0.4:  # 匹配质量阈值
            # 计算与最近添加的模板的角度差异
            angle_diff = abs(np.arctan2(np.sin(current_angle - self.last_template_orientation),
                                        np.cos(current_angle - self.last_template_orientation)))
            return angle_diff > self.min_template_angle_diff

        return False

    def add_new_template(self, frame, bbox, angle, match_quality):
        """添加新模板到模板库

        Args:
            frame: 当前视频帧
            bbox: 目标边界框 (x, y, w, h)
            angle: 当前朝向角度
            match_quality: 匹配质量指标
        """
        # 提取当前目标区域
        x, y, w, h = bbox
        expansion = self.config.get('template_expansion', 20)

        # 扩展边界框
        ex_x = max(0, x - expansion)
        ex_y = max(0, y - expansion)
        ex_w = min(frame.shape[1] - ex_x, w + 2 * expansion)
        ex_h = min(frame.shape[0] - ex_y, h + 2 * expansion)

        # 提取当前ROI
        roi = frame[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w].copy()

        # 提取特征
        features = self.feature_extractor.extract_features(roi)

        # 添加到模板库
        self.template_library.append({
            'roi': roi,
            'features': features,
            'orientation': angle,
            'usage_count': 0,
            'quality': match_quality
        })

        self.last_template_orientation = angle

        # 限制模板库大小
        if len(self.template_library) > 36:
            # 移除使用次数最少的模板(排除初始模板)
            min_usage = float('inf')
            min_idx = -1

            for i, template in enumerate(self.template_library[1:], 1):
                if template['usage_count'] < min_usage:
                    min_usage = template['usage_count']
                    min_idx = i

            if min_idx > 0:
                self.template_library.pop(min_idx)

        print(
            f"添加新模板，角度: {np.degrees(angle):.1f}°, 匹配质量: {match_quality:.2f}, 模板库大小: {len(self.template_library)}")

    def initialize_tracking_variables(self):
        """初始化追踪相关变量"""
        # 初始化轨迹相关变量 - 使用高效数据结构
        self.trajectory = EfficientTrajectory(
            max_points=self.config.get('max_trajectory_points', 1000)
        )
        self.prev_position = None  # 前一帧的位置
        self.direction_angles = []  # 存储朝向角度，用于平滑处理

        # 初始化模板库 - 存储多个不同角度的模板
        self.template_library = []
        self.last_template_orientation = None
        self.min_template_angle_diff = np.radians(10)  # 新模板最小角度差异(10度)

        # 初始化连续低质量匹配计数器
        self.low_quality_match_count = 0

        # 确保模板存储目录存在
        os.makedirs(self.config.get('template_save_path', 'templates'), exist_ok=True)

        # 加载已保存的模板
        if self.config.get('load_saved_templates', False):
            self.load_saved_templates()

        # 初始化GPU加速器（如果可用）
        if self.config.get('use_gpu_acceleration', False):
            self.gpu_transform = GPUAcceleratedTransforms()

        # 初始化轨迹与方向的组合列表
        self.trajectory_with_direction = []

        # 初始化卡尔曼滤波器
        self.has_angle = False
        if self.config.get('use_kalman_filter', True):
            self.initialize_kalman_filter()

    def initialize_annotation_variables(self):
        """初始化标注相关变量"""
        # 多边形标注
        self.current_polygon_points = []  # 当前正在绘制的多边形点
        self.polygons = []  # 已完成的多边形列表 (全局坐标系)

        # 圆形标注
        self.circles = []  # 圆形列表 (全局坐标系)

        # 圆形绘制的临时数据
        self.circle_start_point = None  # 圆心
        self.current_circle_radius = None  # 当前圆半径
        self.temp_circle = None  # 临时圆 (center, radius)
        self.is_drawing_circle = False  # 标记是否正在拖动绘制圆形

    def create_watermark(self):
        """创建水印图像"""
        watermark_text = self.config['watermark_text']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)

        # 创建一个透明的水印图像，大小与帧大小相同
        self.watermark = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # 在水印图像上重复绘制水印文字
        for y in range(0, self.frame_height, text_size[1] * 3):
            for x in range(0, self.frame_width, text_size[0] * 3):
                cv2.putText(self.watermark, watermark_text, (x, y + text_size[1]),
                            font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
                cv2.putText(self.watermark, watermark_text, (x + 1, y + text_size[1] + 1),
                            font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
                cv2.putText(self.watermark, watermark_text, (x - 1, y + text_size[1] - 1),
                            font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # 生成水印掩码
        self.watermark_mask = cv2.cvtColor(self.watermark, cv2.COLOR_BGR2GRAY)
        _, self.watermark_mask = cv2.threshold(self.watermark_mask, 1, 255, cv2.THRESH_BINARY)

    def estimate_orientation_by_features(self, frame, bbox, expansion=10):
        """使用特征匹配估计目标朝向 - 动态多模板版

        Args:
            frame: 当前帧
            bbox: 目标边界框 (x, y, w, h)
            expansion: 边界框扩展像素数

        Returns:
            tuple: (朝向角度(弧度), 可视化数据)
        """
        # 如果还没有模板特征，回退到矩形拟合方法
        if not hasattr(self, 'template_library') or not self.template_library:
            return self.estimate_orientation_by_rectangle(frame, bbox, expansion)

        x, y, w, h = bbox

        # 扩展边界框
        ex_x = max(0, x - expansion)
        ex_y = max(0, y - expansion)
        ex_w = min(frame.shape[1] - ex_x, w + 2 * expansion)
        ex_h = min(frame.shape[0] - ex_y, h + 2 * expansion)

        # 提取当前ROI区域
        roi = frame[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w].copy()

        try:
            # 提取当前ROI的特征
            roi_features = self.feature_extractor.extract_features(roi)

            # 预估朝向 - 使用上一帧的朝向或默认朝向
            estimated_orientation = self.direction_angles[-1] if self.direction_angles else self.initial_orientation

            # 查找最佳匹配模板
            best_template_idx = self.find_best_template(estimated_orientation)
            best_template = self.template_library[best_template_idx]

            # 特征匹配
            match_result = self.feature_extractor.match_features(best_template['features'], roi_features)

            # 获取SIFT匹配结果
            good_matches = match_result.get('sift_matches', [])

            # 检查CNN相似度
            cnn_similarity = match_result.get('cnn_similarity')

            if self.config.get('debug_feature_matching', False) and cnn_similarity is not None:
                print(f"CNN特征相似度: {cnn_similarity:.4f}, 模板角度: {np.degrees(best_template['orientation']):.1f}°")

            # 特征点太少，回退到矩形拟合或触发校准
            if len(good_matches) < 8:  # 最小需要8点来稳定估计仿射变换
                if self.config.get('debug_feature_matching', False):
                    debug_img = np.zeros((100, 400, 3), dtype=np.uint8)
                    text = f"匹配点不足: 仅有{len(good_matches)}点"
                    cv2.putText(debug_img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Feature Matching', debug_img)

                # 直接触发校准，不再累计计数
                if self.current_state == State.PLAYING and not self.config.get('auto_continue_on_poor_quality', False):
                    print("检测到匹配点不足，暂停并准备手动校准...")
                    prev_state = self.current_state
                    self.current_state = State.PAUSED
                    self.manual_orientation_calibration()

                    # 使用校准后的朝向值直接返回
                    if self.direction_angles:
                        calibrated_angle = self.direction_angles[-1]
                        # 创建简单的可视化数据以保持返回格式一致
                        center = (int(x + w / 2), int(y + h / 2))
                        arrow_length = 30
                        arrow_end = (
                            center[0] + int(arrow_length * np.cos(calibrated_angle)),
                            center[1] + int(arrow_length * np.sin(calibrated_angle))
                        )

                        # 恢复之前的状态
                        self.current_state = prev_state

                        vis_data = {
                            'center': center,
                            'arrow_end': arrow_end,
                            'inlier_ratio': 1.0,  # 校准后可信度视为最高
                            'cnn_similarity': 1.0,
                            'template_idx': best_template_idx,
                            'match_quality': 1.0
                        }
                        return calibrated_angle, vis_data

                # 如果设置为自动继续或校准后返回，回退到矩形拟合
                return self.estimate_orientation_by_rectangle(frame, bbox, expansion)

            # 获取匹配的关键点坐标
            template_keypoints = best_template['features']['keypoints']
            src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([roi_features['keypoints'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用RANSAC算法估计仿射变换矩阵
            M, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.get('ransac_threshold', 3.0),
                confidence=self.config.get('ransac_confidence', 0.99),
                maxIters=self.config.get('ransac_iterations', 2000)
            )

            # 如果变换矩阵估计失败，回退到矩形拟合或触发校准
            if M is None:
                if self.config.get('debug_feature_matching', False):
                    debug_img = np.zeros((100, 400, 3), dtype=np.uint8)
                    text = "仿射变换估计失败"
                    cv2.putText(debug_img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Feature Matching', debug_img)

                # 直接触发校准，不再累计计数
                if self.current_state == State.PLAYING and not self.config.get('auto_continue_on_poor_quality', False):
                    print("仿射变换估计失败，暂停并准备手动校准...")
                    prev_state = self.current_state
                    self.current_state = State.PAUSED
                    self.manual_orientation_calibration()

                    # 使用校准后的朝向返回
                    if self.direction_angles:
                        calibrated_angle = self.direction_angles[-1]
                        center = (int(x + w / 2), int(y + h / 2))
                        arrow_length = 30
                        arrow_end = (
                            center[0] + int(arrow_length * np.cos(calibrated_angle)),
                            center[1] + int(arrow_length * np.sin(calibrated_angle))
                        )

                        # 恢复之前的状态
                        self.current_state = prev_state

                        vis_data = {
                            'center': center,
                            'arrow_end': arrow_end,
                            'inlier_ratio': 1.0,
                            'cnn_similarity': 1.0,
                            'template_idx': best_template_idx,
                            'match_quality': 1.0
                        }
                        return calibrated_angle, vis_data

                # 如果设置为自动继续或校准后返回，回退到矩形拟合
                return self.estimate_orientation_by_rectangle(frame, bbox, expansion)

            # 统计内点数量
            inlier_count = np.sum(mask) if mask is not None else 0
            inlier_ratio = inlier_count / len(good_matches) if len(good_matches) > 0 else 0

            # 从仿射矩阵中提取旋转角度
            angle_rad = np.arctan2(M[1, 0], M[0, 0])

            # 计算当前朝向 - 基于模板的原始朝向加上变化量
            current_orientation = best_template['orientation'] + angle_rad

            # 计算匹配质量
            match_quality = inlier_ratio * min(len(good_matches) / 20, 1.0)

            # 检测角度跳变
            if self.direction_angles and len(self.direction_angles) > 0:
                last_angle = self.direction_angles[-1]
                angle_diff = abs(
                    np.arctan2(np.sin(current_orientation - last_angle), np.cos(current_orientation - last_angle)))

                # 如果角度变化超过配置的阈值且匹配质量不高，认为是角度跳变
                jump_threshold = np.radians(self.config.get('orientation_jump_threshold', 45))
                if angle_diff > jump_threshold and match_quality < 0.7 and self.current_state == State.PLAYING:
                    if not self.config.get('auto_continue_on_poor_quality', False):
                        print(f"检测到角度跳变: {np.degrees(angle_diff):.1f}°，暂停并准备手动校准...")
                        prev_state = self.current_state
                        self.current_state = State.PAUSED
                        self.manual_orientation_calibration()

                        # 使用校准后的朝向
                        if self.direction_angles:
                            calibrated_angle = self.direction_angles[-1]
                            center = (int(x + w / 2), int(y + h / 2))
                            arrow_length = 30
                            arrow_end = (
                                center[0] + int(arrow_length * np.cos(calibrated_angle)),
                                center[1] + int(arrow_length * np.sin(calibrated_angle))
                            )

                            # 恢复之前的状态
                            self.current_state = prev_state

                            vis_data = {
                                'center': center,
                                'arrow_end': arrow_end,
                                'inlier_ratio': 1.0,
                                'cnn_similarity': 1.0,
                                'template_idx': best_template_idx,
                                'match_quality': 1.0
                            }
                            return calibrated_angle, vis_data
                    else:
                        print(f"检测到角度跳变: {np.degrees(angle_diff):.1f}°，但已配置自动继续...")
                        # 使用上一帧的朝向角度降低突变
                        if len(self.direction_angles) > 0:
                            # 将当前朝向角度朝上一帧角度靠近，减少突变
                            smoothed_angle = last_angle + (current_orientation - last_angle) * 0.2
                            return smoothed_angle, None

            # 检查匹配质量，质量低于阈值触发校准
            min_quality = self.config.get('min_match_quality', 0.4)
            if match_quality < min_quality:
                # 直接触发校准
                if self.current_state == State.PLAYING and not self.config.get('auto_continue_on_poor_quality', False):
                    print(f"检测到低质量匹配: {match_quality:.2f}，暂停并准备手动校准...")
                    prev_state = self.current_state
                    self.current_state = State.PAUSED
                    self.manual_orientation_calibration()

                    # 使用校准后的朝向
                    if self.direction_angles:
                        calibrated_angle = self.direction_angles[-1]
                        center = (int(x + w / 2), int(y + h / 2))
                        arrow_length = 30
                        arrow_end = (
                            center[0] + int(arrow_length * np.cos(calibrated_angle)),
                            center[1] + int(arrow_length * np.sin(calibrated_angle))
                        )

                        # 恢复之前的状态
                        self.current_state = prev_state

                        vis_data = {
                            'center': center,
                            'arrow_end': arrow_end,
                            'inlier_ratio': 1.0,
                            'cnn_similarity': 1.0,
                            'template_idx': best_template_idx,
                            'match_quality': 1.0
                        }
                        return calibrated_angle, vis_data

                elif self.config.get('auto_continue_on_poor_quality', False):
                    print(f"检测到低质量匹配: {match_quality:.2f}，但已配置自动继续...")

            # 判断是否需要添加新模板 - 只有在匹配质量高的情况下才添加，避免添加错误模板
            high_quality_threshold = 0.6  # 添加新模板的质量阈值
            if self.should_add_new_template(current_orientation,
                                            match_quality) and match_quality > high_quality_threshold:
                self.add_new_template(frame, bbox, current_orientation, match_quality)

            # 创建用于可视化的数据
            # 从变换矩阵计算模板中心在当前ROI中的位置
            template_center = np.array([best_template['roi'].shape[1] / 2, best_template['roi'].shape[0] / 2, 1])
            transformed_center = np.matmul(M, template_center)

            # 调整中心点和箭头终点的坐标到原始图像坐标系
            center = (int(ex_x + transformed_center[0]), int(ex_y + transformed_center[1]))

            # 计算旋转后的箭头终点
            arrow_length = 30
            arrow_end_x = center[0] + int(arrow_length * np.cos(current_orientation))
            arrow_end_y = center[1] + int(arrow_length * np.sin(current_orientation))
            arrow_end = (arrow_end_x, arrow_end_y)

            # 返回朝向角度和可视化数据
            vis_data = {
                'center': center,
                'arrow_end': arrow_end,
                'inlier_ratio': inlier_ratio,
                'cnn_similarity': cnn_similarity,
                'template_idx': best_template_idx,
                'match_quality': match_quality
            }

            return current_orientation, vis_data

        except Exception as e:
            print(f"特征匹配朝向估计出错: {e}")
            import traceback
            traceback.print_exc()

            # 直接触发校准
            if self.current_state == State.PLAYING and not self.config.get('auto_continue_on_poor_quality', False):
                print("检测到异常，暂停并准备手动校准...")
                prev_state = self.current_state
                self.current_state = State.PAUSED
                self.manual_orientation_calibration()

                # 使用校准后的朝向
                if self.direction_angles:
                    calibrated_angle = self.direction_angles[-1]
                    center = (int(x + w / 2), int(y + h / 2))
                    arrow_length = 30
                    arrow_end = (
                        center[0] + int(arrow_length * np.cos(calibrated_angle)),
                        center[1] + int(arrow_length * np.sin(calibrated_angle))
                    )

                    # 恢复之前的状态
                    self.current_state = prev_state

                    vis_data = {
                        'center': center,
                        'arrow_end': arrow_end,
                        'inlier_ratio': 1.0,
                        'cnn_similarity': None,
                        'template_idx': 0,
                        'match_quality': 1.0
                    }
                    return calibrated_angle, vis_data

            return self.estimate_orientation_by_rectangle(frame, bbox, expansion)

    def estimate_orientation_by_rectangle(self, frame, bbox, expansion=10):
        """基于矩形拟合的朝向估计方法

        Args:
            frame: 当前帧
            bbox: 目标边界框 (x, y, w, h)
            expansion: 边界框扩展像素数

        Returns:
            tuple: (朝向角度(弧度), 矩形角点)
        """
        x, y, w, bbox_height = bbox  # 重命名为bbox_height避免命名冲突
        # 扩展边界框 - 减小扩展范围更聚焦于车辆本身
        ex_x = max(0, x - expansion // 2)
        ex_y = max(0, y - expansion // 2)
        ex_w = min(frame.shape[1] - ex_x, w + expansion)
        ex_h = min(frame.shape[0] - ex_y, bbox_height + expansion)

        # 提取ROI区域
        roi = frame[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w]

        # 转换到HSV色彩空间以更好地分离白色车辆
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 提取亮度高的区域（针对白色车辆）
        # 使用不同的变量名避免冲突
        hue, saturation, value = cv2.split(hsv)

        # 设置提取白色区域的阈值范围 - 白色通常有低饱和度和高亮度
        # 低饱和度掩码
        _, s_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY_INV)  # 从30调整到50
        _, v_mask = cv2.threshold(value, 180, 255, cv2.THRESH_BINARY)  # 从200调整到180
        # 合并掩码找到同时满足低饱和度和高亮度的区域（白色）
        white_mask = cv2.bitwise_and(s_mask, v_mask)

        # 形态学操作增强车辆轮廓
        kernel = np.ones((8, 8), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)  # 去除小噪点
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)  # 填充小洞

        # 查找轮廓
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建调试可视化
        if self.config.get('debug_orientation', False):
            debug_roi = roi.copy()
            cv2.drawContours(debug_roi, contours, -1, (0, 255, 0), 2)
            white_mask_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
            hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            debug_img = np.hstack([roi, hsv_vis, white_mask_vis, debug_roi])
            cv2.imshow('Orientation Debug', debug_img)

        if not contours:
            return None, None

        # 过滤太小的轮廓 - 使用bbox_height而不是h
        min_area = w * bbox_height * 0.05  # 至少为边界框面积的5%

        # 添加调试信息
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not valid_contours:
            return None, None

        # 选择最大的轮廓
        largest_contour = max(valid_contours, key=cv2.contourArea)
        # 计算最小面积矩形
        rect = cv2.minAreaRect(largest_contour)
        center, (width, height), angle = rect

        # 调整中心点坐标到原始图像坐标系
        center = (ex_x + center[0], ex_y + center[1])

        # 获取矩形的四个角点（用于可视化）
        box_points = cv2.boxPoints(rect)
        box_points = np.intp(box_points)
        # 调整角点坐标到原始图像坐标系
        box_points[:, 0] += ex_x
        box_points[:, 1] += ex_y

        # 将OpenCV角度转换为弧度
        # OpenCV的minAreaRect返回的角度范围是[-90, 0)
        angle_rad = np.radians(angle)

        # 确保朝向总是指向长边
        if width < height:
            # 如果宽度小于高度，表示长边与y轴更接近，调整角度加90°
            angle_rad += np.pi / 2

        # 解决长边朝向的歧义性（长边有两个可能的方向）
        # 使用速度信息确定具体指向哪个方向
        if self.prev_position is not None and len(self.trajectory) > 1:
            # 计算当前速度向量
            current_pos = self.trajectory[-1]
            prev_pos = self.trajectory[-2]
            velocity_vector = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])

            # 避免静止状态下的不稳定
            velocity_magnitude = np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2)
            if velocity_magnitude > 0.5:  # 速度阈值
                # 计算速度向量的角度
                velocity_angle = np.arctan2(velocity_vector[1], velocity_vector[0])

                # 计算矩形两个可能方向与速度方向的夹角
                angle1 = angle_rad
                angle2 = angle_rad + np.pi  # 相反方向

                # 计算两个角度与速度角度的夹角差异
                diff1 = abs(np.arctan2(np.sin(angle1 - velocity_angle), np.cos(angle1 - velocity_angle)))
                diff2 = abs(np.arctan2(np.sin(angle2 - velocity_angle), np.cos(angle2 - velocity_angle)))

                # 确保diff1和diff2是标量
                if isinstance(diff1, np.ndarray):
                    diff1 = float(diff1.item())
                if isinstance(diff2, np.ndarray):
                    diff2 = float(diff2.item())

                # 选择与速度方向夹角较小的朝向
                if diff1 > diff2:
                    angle_rad = angle2

        return angle_rad, box_points

    def initialize_kalman_filter(self):
        """初始化卡尔曼滤波器用于朝向平滑"""
        # 状态变量: [角度, 角速度]
        self.kalman = cv2.KalmanFilter(2, 1)
        self.kalman.measurementMatrix = np.array([[1.0, 0.0]], np.float32)

        # 转移矩阵
        self.kalman.transitionMatrix = np.array([[1.0, 1.0],
                                                 [0.0, 1.0]], np.float32)

        # 过程噪声协方差
        self.kalman.processNoiseCov = np.array([[0.01, 0.0],
                                                [0.0, 0.01]], np.float32)

        # 测量噪声协方差 - 默认值
        self.kalman.measurementNoiseCov = np.array([[0.1]], np.float32)

        # 初始状态及其协方差
        self.has_angle = False  # 标记是否已初始化

    def update_angle_with_kalman(self, new_angle, confidence=1.0):
        """使用卡尔曼滤波器更新角度，考虑测量可信度

        Args:
            new_angle: 新测量的角度（弧度）
            confidence: 测量的可信度（0-1）

        Returns:
            float: 滤波后的角度（弧度）
        """
        if not hasattr(self, 'kalman') or self.kalman is None:
            self.initialize_kalman_filter()

        # 基于置信度调整测量噪声 - 低置信度时提高噪声（减少对测量的信任）
        noise_scale = 1.0 / max(confidence, 0.1)  # 避免除零
        self.kalman.measurementNoiseCov = np.array([[0.1 * noise_scale]], np.float32)

        # 初始化状态
        if not self.has_angle:
            self.kalman.statePre = np.array([[new_angle], [0]], np.float32)
            self.kalman.statePost = np.array([[new_angle], [0]], np.float32)
            self.has_angle = True
            return new_angle

        # 角度需要特殊处理，确保在[-π,π]范围内平滑变化
        predicted = self.kalman.predict()
        pred_angle = predicted[0, 0]

        # 解决角度跳变问题（-π到π的循环）
        angle_diff = new_angle - pred_angle
        if angle_diff > np.pi:
            new_angle -= 2 * np.pi
        elif angle_diff < -np.pi:
            new_angle += 2 * np.pi

        # 大于90度的变化可能是异常值，检查是否应该拒绝
        if abs(angle_diff) > np.pi / 3 and confidence < 0.8:
            print(f"拒绝异常角度变化: {np.degrees(angle_diff):.1f}°，置信度: {confidence:.2f}")
            return pred_angle

        # 更新测量值
        measurement = np.array([[new_angle]], np.float32)
        corrected = self.kalman.correct(measurement)

        return corrected[0, 0]

    def transform_point(self, point, matrix):
        """将点从一个坐标系转换到另一个坐标系

        Args:
            point (tuple): 输入点坐标 (x, y)
            matrix (ndarray): 转换矩阵

        Returns:
            tuple: 转换后的点坐标 (x, y)
        """
        if self.config.get('use_gpu_acceleration', False) and hasattr(self, 'gpu_transform'):
            # 使用GPU加速
            transformed = self.gpu_transform.transform_points_gpu([point], matrix)[0]
            return (transformed[0], transformed[1])
        else:
            # 使用原始的CPU实现
            point_array = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point_array, matrix)
            return (transformed[0, 0][0], transformed[0, 0][1])


    def transform_points(self, points, matrix):
        """将多个点从一个坐标系转换到另一个坐标系

        Args:
            points (list): 输入点列表 [(x1, y1), (x2, y2), ...]
            matrix (ndarray): 转换矩阵

        Returns:
            list: 转换后的点列表
        """
        if len(points) == 0:
            return []

        if self.config.get('use_gpu_acceleration', False) and hasattr(self, 'gpu_transform'):
            # 使用GPU加速
            return self.gpu_transform.transform_points_gpu(points, matrix)
        else:
            # 使用原始的CPU实现
            points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(points_array, matrix)
            return transformed.reshape(-1, 2).tolist()

    def clear_current_drawing(self):
        """清除当前正在绘制但未保存的标注"""
        self.current_polygon_points = []
        self.circle_start_point = None
        self.current_circle_radius = None
        self.temp_circle = None
        self.is_drawing_circle = False  # 重置绘制状态

    def mouse_callback(self, event, x, y, flags, param):
        """主窗口的鼠标事件处理函数"""
        if self.current_state == State.DRAWING_POLYGON:
            self.handle_mouse_polygon(event, x, y)
        elif self.current_state == State.DRAWING_CIRCLE:
            self.handle_mouse_circle(event, x, y)

    def handle_mouse_polygon(self, event, x, y):
        """处理多边形绘制过程中的鼠标事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加多边形点
            self.current_polygon_points.append((x, y))
            # 更新显示
            self.update_display()

    def handle_mouse_circle(self, event, x, y):
        """处理圆形绘制过程中的鼠标事件，实现直观的拖动绘制效果"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制圆形，设置圆心
            self.circle_start_point = (x, y)
            self.current_circle_radius = 0
            self.temp_circle = None
            # 标记正在拖动绘制圆形的状态
            self.is_drawing_circle = True

        elif event == cv2.EVENT_MOUSEMOVE and self.circle_start_point is not None and self.is_drawing_circle:
            # 只有在按住鼠标左键拖动状态下才更新圆形
            dx = x - self.circle_start_point[0]
            dy = y - self.circle_start_point[1]
            self.current_circle_radius = int(np.sqrt(dx * dx + dy * dy))
            self.update_display()

        elif event == cv2.EVENT_LBUTTONUP and self.circle_start_point is not None and self.is_drawing_circle:
            # 鼠标释放时固定圆形
            dx = x - self.circle_start_point[0]
            dy = y - self.circle_start_point[1]
            self.current_circle_radius = int(np.sqrt(dx * dx + dy * dy))
            # 保存为临时圆形，等待确认或取消
            self.temp_circle = (self.circle_start_point, self.current_circle_radius)
            # 结束拖动状态
            self.is_drawing_circle = False
            print("圆形已固定，按Enter保存或Backspace取消")
            self.update_display()

    def update_display(self):
        """更新显示帧"""
        if self.current_frame is not None:
            self.display_frame = self.current_frame.copy()

            # 根据当前状态决定显示内容
            if self.current_state == State.DRAWING_POLYGON:
                # 显示当前多边形点和连线
                if len(self.current_polygon_points) > 0:
                    # 画点
                    for point in self.current_polygon_points:
                        cv2.circle(self.display_frame, point, 3, self.config['polygon_point_color'], -1)

                    # 连线 - 保持开放状态 (不闭合)
                    if len(self.current_polygon_points) > 1:
                        points = np.array(self.current_polygon_points, dtype=np.int32)
                        cv2.polylines(self.display_frame, [points], False, self.config['polygon_line_color'], 2)

            elif self.current_state == State.DRAWING_CIRCLE:
                # 显示当前正在绘制的圆形（带填充）
                if self.circle_start_point is not None and self.current_circle_radius is not None:
                    # 创建圆形掩码
                    mask = np.zeros_like(self.display_frame)
                    cv2.circle(mask, self.circle_start_point, self.current_circle_radius,
                               self.config['circle_fill_color'], -1)

                    # 应用透明度
                    alpha = self.config['circle_alpha']
                    cv2.addWeighted(self.display_frame, 1, mask, alpha, 0, self.display_frame)

                    # 绘制边框
                    cv2.circle(self.display_frame, self.circle_start_point, self.current_circle_radius,
                               self.config['circle_border_color'],
                               self.config['circle_border_thickness'])

            # 渲染已保存的标注
            self.render_annotations(self.display_frame)

            # 显示FPS（如果启用）
            if self.config.get('show_fps', False) and len(self.frame_times) > 0:
                avg_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                cv2.putText(self.display_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 显示
            cv2.imshow(self.main_window_name, self.display_frame)

    def handle_key(self, key):
        """处理键盘事件 - 添加导出功能"""
        if key == 27:  # ESC键
            return False  # 退出循环

        # 根据当前状态处理不同的键盘事件
        if key == 32:  # 空格键
            self.handle_space_key()
        elif key == ord('p'):  # p键
            self.handle_p_key()
        elif key == ord('o'):  # o键
            self.handle_o_key()
        elif key == ord('u'):  # u键 - 新增朝向校准
            self.handle_u_key()
        elif key == ord('e'):  # 新增：e键导出数据
            self.handle_e_key()
        elif key == 13:  # Enter键
            self.handle_enter_key()
        elif key == 8:  # Backspace键
            self.handle_backspace_key()

        return True  # 继续循环

    def handle_u_key(self):
        """处理u键事件 - 手动校准朝向"""
        if self.current_state == State.PAUSED:
            print("进入手动朝向校准模式...")
            self.manual_orientation_calibration()
        else:
            print("请先按空格键暂停视频，再按'u'键进行朝向校准")

    def handle_e_key(self):
        """处理e键事件 - 导出追踪数据"""
        if self.current_state == State.PAUSED:
            export_path = self.config.get('tracking_data_export_path', 'tracking_data_5_29_good.json')
            self.export_tracking_data(export_path)
        else:
            print("请先按空格键暂停视频，再按'e'键导出数据")

    def manual_orientation_calibration(self):
        """手动校准朝向"""
        # 获取当前跟踪的边界框
        success, bbox = self.tracker.update(self.current_frame.copy())
        if not success:
            print("找不到跟踪目标，无法校准朝向")
            return

        # 提取ROI区域，展示给用户校准朝向
        x, y, w, h = [int(v) for v in bbox]

        # 提取目标区域作为模板，增加一些外边距
        expansion = self.config.get('template_expansion', 10)
        ex_x = max(0, x - expansion // 2)
        ex_y = max(0, y - expansion // 2)
        ex_w = min(self.current_frame.shape[1] - ex_x, w + expansion)
        ex_h = min(self.current_frame.shape[0] - ex_y, h + expansion)

        template_roi = self.original_frame[ex_y:ex_y + ex_h, ex_x:ex_x + ex_w].copy()

        # 显示方向校准窗口
        orientation_window = "手动校准朝向"
        cv2.namedWindow(orientation_window)

        # 初始化变量
        center_x, center_y = ex_w // 2, ex_h // 2
        arrow_length = 30

        # 如果有最近的方向角度，使用它作为初始值
        if self.direction_angles:
            last_angle = self.direction_angles[-1]
            current_angle = np.degrees(last_angle)
        else:
            current_angle = -90  # 默认朝上

        angle_step = 5  # 初始步长为5度

        # 计算初始箭头终点
        angle_rad = np.radians(current_angle)
        direction_x = center_x + int(arrow_length * np.cos(angle_rad))
        direction_y = center_y + int(arrow_length * np.sin(angle_rad))

        print("\n请使用键盘校准朝向:")
        print("←/A: 逆时针旋转箭头")
        print("→/D: 顺时针旋转箭头")
        print("↑: 增加旋转步长")
        print("↓: 减小旋转步长")
        print("Enter: 确认朝向")
        print("ESC: 取消校准")

        while True:
            # 创建显示图像
            display = template_roi.copy()

            # 绘制箭头和说明
            cv2.line(display, (center_x, center_y), (direction_x, direction_y), (0, 255, 0), 2)
            cv2.putText(display, "使用左右方向键或A/D键调整角度", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display, "完成后按回车确认", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 绘制当前角度和步长
            cv2.putText(display, f"当前角度: {current_angle:.1f}°", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(display, f"步长: {angle_step:.1f}°", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 显示图像
            cv2.imshow(orientation_window, display)

            # 获取键盘输入
            key = cv2.waitKey(20) & 0xFF

            # 如果按下回车，确认朝向并退出
            if key == 13:  # 回车键
                break

            # 如果按下ESC，取消校准并退出
            if key == 27:  # ESC键
                cv2.destroyWindow(orientation_window)
                return

            # 根据按键调整方向
            if key == ord('a') or key == 81:  # A 或 左箭头: 逆时针旋转
                current_angle -= angle_step
            elif key == ord('d') or key == 83:  # D 或 右箭头: 顺时针旋转
                current_angle += angle_step
            elif key == 82:  # 上箭头: 增加步长
                angle_step = min(angle_step + 1, 45)  # 最大步长45度
            elif key == 84:  # 下箭头: 减小步长
                angle_step = max(angle_step - 1, 1)  # 最小步长1度

            # 限制角度在[-180, 180]范围内
            current_angle = ((current_angle + 180) % 360) - 180

            # 更新箭头终点坐标
            angle_rad = np.radians(current_angle)
            direction_x = center_x + int(arrow_length * np.cos(angle_rad))
            direction_y = center_y + int(arrow_length * np.sin(angle_rad))

        # 计算校准后的朝向角度（弧度制）
        calibrated_orientation = np.radians(current_angle)

        # 提取模板特征
        features = self.feature_extractor.extract_features(template_roi)

        # 添加到模板库
        self.add_calibrated_template(template_roi, features, calibrated_orientation)

        # 保存模板到磁盘
        self.save_template(template_roi, calibrated_orientation)

        # 重置卡尔曼滤波器以适应新的朝向
        if self.config.get('use_kalman_filter', True):
            self.has_angle = False
            self.kalman.statePre = np.array([[calibrated_orientation], [0]], np.float32)
            self.kalman.statePost = np.array([[calibrated_orientation], [0]], np.float32)
            self.has_angle = True

        # 更新最近的方向角度
        if self.direction_angles:
            self.direction_angles[-1] = calibrated_orientation

        # 重置低质量匹配计数器
        self.low_quality_match_count = 0

        # 关闭校准窗口
        cv2.destroyWindow(orientation_window)

        print(f"朝向已校准为: {current_angle:.1f}°")

    def add_calibrated_template(self, roi, features, orientation):
        """添加手动校准的模板到模板库"""
        # 添加到模板库，并标记为高质量手动校准模板
        self.template_library.append({
            'roi': roi.copy(),
            'features': features,
            'orientation': orientation,
            'usage_count': 0,
            'quality': 1.0,  # 最高质量
            'manual_calibrated': True  # 标记为手动校准
        })

        self.last_template_orientation = orientation
        print(f"添加手动校准模板，角度: {np.degrees(orientation):.1f}°")

    def save_template(self, roi, orientation):
        """保存模板到磁盘"""
        try:
            import csv

            # 创建唯一的文件名
            timestamp = int(time.time())
            angle_deg = int(np.degrees(orientation))
            filename = f"template_{timestamp}_{angle_deg}.png"
            filepath = os.path.join(self.config['template_save_path'], filename)

            # 保存图像
            cv2.imwrite(filepath, roi)

            # 保存元数据
            metadata_file = os.path.join(self.config['template_save_path'], "metadata.csv")
            file_exists = os.path.isfile(metadata_file)

            with open(metadata_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['filename', 'orientation', 'timestamp'])
                writer.writerow([filename, orientation, timestamp])

            print(f"模板已保存到: {filepath}")
        except Exception as e:
            print(f"保存模板失败: {e}")

    def load_saved_templates(self):
        """从磁盘加载已保存的模板"""
        import csv

        metadata_file = os.path.join(self.config['template_save_path'], "metadata.csv")
        if not os.path.isfile(metadata_file):
            print("没有找到已保存的模板元数据")
            return

        try:
            loaded_count = 0
            with open(metadata_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row['filename']
                    orientation = float(row['orientation'])

                    filepath = os.path.join(self.config['template_save_path'], filename)
                    if os.path.isfile(filepath):
                        # 加载模板图像
                        template_roi = cv2.imread(filepath)
                        if template_roi is None:
                            continue

                        # 提取特征
                        features = self.feature_extractor.extract_features(template_roi)

                        # 添加到模板库
                        self.template_library.append({
                            'roi': template_roi,
                            'features': features,
                            'orientation': orientation,
                            'usage_count': 0,
                            'quality': 1.0,  # 高质量
                            'manual_calibrated': True  # 标记为手动校准
                        })
                        loaded_count += 1

            if loaded_count > 0:
                print(f"已加载 {loaded_count} 个已保存的模板")
        except Exception as e:
            print(f"加载模板失败: {e}")

    def handle_space_key(self):
        """处理空格键事件"""
        if self.current_state == State.PLAYING:
            self.current_state = State.PAUSED
            print("视频已暂停，按 'p' 绘制多边形，按 'o' 绘制圆形，按空格继续播放")
        elif self.current_state == State.PAUSED:
            self.current_state = State.PLAYING
            print("继续播放视频")
        elif self.current_state in [State.DRAWING_POLYGON, State.DRAWING_CIRCLE]:
            # 放弃当前绘制
            self.clear_current_drawing()
            self.current_state = State.PLAYING
            print("放弃当前绘制，继续播放视频")

    def handle_p_key(self):
        """处理p键事件"""
        if self.current_state == State.PAUSED:
            # 进入多边形绘制模式，清除任何现有的未保存标注
            self.clear_current_drawing()
            self.current_state = State.DRAWING_POLYGON
            print("进入多边形绘制模式，点击添加顶点，按Enter保存，按Backspace撤销点")
        elif self.current_state == State.DRAWING_CIRCLE:
            # 从圆形模式切换到多边形模式
            self.clear_current_drawing()
            self.current_state = State.DRAWING_POLYGON
            print("切换到多边形绘制模式，点击添加顶点，按Enter保存，按Backspace撤销点")

    def handle_o_key(self):
        """处理o键事件"""
        if self.current_state == State.PAUSED:
            # 进入圆形绘制模式，清除任何现有的未保存标注
            self.clear_current_drawing()
            self.current_state = State.DRAWING_CIRCLE
            print("进入圆形绘制模式，拖动鼠标绘制圆形，释放确认，按Enter保存，按Backspace撤销")
        elif self.current_state == State.DRAWING_POLYGON:
            # 从多边形模式切换到圆形模式
            self.clear_current_drawing()
            self.current_state = State.DRAWING_CIRCLE
            print("切换到圆形绘制模式，拖动鼠标绘制圆形，释放确认，按Enter保存，按Backspace撤销")

    def handle_enter_key(self):
        """处理Enter键事件"""
        if self.current_state == State.DRAWING_POLYGON:
            self.save_current_polygon()
        elif self.current_state == State.DRAWING_CIRCLE:
            self.save_current_circle()

    def save_current_polygon(self):
        """保存当前多边形 - 简化版"""
        if len(self.current_polygon_points) >= 3:
            # 将多边形从当前帧坐标系转换到全局坐标系
            global_polygon = self.transform_points(self.current_polygon_points, self.cumulative_transform)

            # 添加到多边形列表
            self.polygons.append(global_polygon)

            # 重置
            self.current_polygon_points = []

            # 返回暂停状态
            self.current_state = State.PAUSED
            print("多边形已保存，按空格继续播放，或按'p'/'o'继续绘制，按'e'导出数据")
        else:
            print("至少需要3个点才能形成多边形")

        # 更新显示
        self.update_display()

        # 更新显示
        self.update_display()

    def save_current_circle(self):
        """保存当前圆形，支持refine模式"""
        if self.temp_circle and self.current_circle_radius > 0:
            center, radius = self.temp_circle

            # 将圆形从当前帧坐标系转换到全局坐标系
            global_center = self.transform_point(center, self.cumulative_transform)

            # 计算全局坐标系中的半径
            scale_factor = np.sqrt(np.abs(np.linalg.det(self.cumulative_transform[:2, :2])))
            global_radius = radius * scale_factor

            # 添加到圆形列表
            self.circles.append((global_center, global_radius))

            # 重置
            self.clear_current_drawing()

            # 返回暂停状态
            self.current_state = State.PAUSED
            print("圆形已保存，按空格继续播放，或按'p'/'o'继续绘制")
        else:
            print("请先创建有效的圆形")

        # 更新显示
        self.update_display()

    def handle_backspace_key(self):
        """处理Backspace键事件"""
        if self.current_state == State.DRAWING_POLYGON and len(self.current_polygon_points) > 0:
            # 移除最后一个点
            self.current_polygon_points.pop()
            print(f"删除最后一个点，剩余 {len(self.current_polygon_points)} 个点")
            self.update_display()
        elif self.current_state == State.DRAWING_CIRCLE and self.temp_circle:
            # 重置圆形绘制
            self.clear_current_drawing()
            print("重置圆形绘制")
            self.update_display()
        elif self.current_state == State.PAUSED:
            # 删除最后添加的标注
            if len(self.circles) > 0:
                self.circles.pop()
                print("删除最后一个圆形")
            elif len(self.polygons) > 0:
                self.polygons.pop()
                print("删除最后一个多边形")

            # 更新显示
            self.update_display()

    def render_annotations(self, frame):
        """原始的标注渲染方法"""
        # 获取逆变换矩阵
        inv_transform = np.linalg.inv(self.cumulative_transform)

        # 渲染所有多边形
        for global_polygon in self.polygons:
            try:
                # 将全局坐标系中的多边形转换到当前帧坐标系
                local_polygon = self.transform_points(global_polygon, inv_transform)
                local_polygon = np.array(local_polygon, dtype=np.int32)

                # 创建掩码和覆盖层
                mask = np.zeros_like(frame)
                cv2.fillPoly(mask, [local_polygon], self.config['polygon_fill_color'])

                # 应用透明度
                alpha = self.config['polygon_alpha']
                cv2.addWeighted(frame, 1, mask, alpha, 0, frame)

                # 绘制边框
                cv2.polylines(frame, [local_polygon], True,
                              self.config['polygon_border_color'],
                              self.config['polygon_border_thickness'])

            except Exception as e:
                print(f"渲染多边形时出错: {e}")

        # 渲染所有圆形
        for center, radius in self.circles:
            try:
                # 将全局坐标系中的圆心转换到当前帧坐标系
                local_center = self.transform_point(center, inv_transform)
                local_center = (int(local_center[0]), int(local_center[1]))

                # 计算变换后的半径
                scale_factor = np.sqrt(np.abs(np.linalg.det(inv_transform[:2, :2])))
                local_radius = int(radius * scale_factor)

                # 创建掩码和覆盖层
                mask = np.zeros_like(frame)
                cv2.circle(mask, local_center, local_radius, self.config['circle_fill_color'], -1)

                # 应用透明度
                alpha = self.config['circle_alpha']
                cv2.addWeighted(frame, 1, mask, alpha, 0, frame)

                # 绘制边框
                cv2.circle(frame, local_center, local_radius,
                           self.config['circle_border_color'],
                           self.config['circle_border_thickness'])

            except Exception as e:
                print(f"渲染圆形时出错: {e}")

    def smooth_direction_angles(self, angles, window_size):
        """计算朝向角度的平滑值"""
        if len(angles) < window_size:
            return angles

        # 对角度进行平滑处理需要特殊处理，因为角度是循环的
        # 将角度转换为单位向量
        x_components = np.cos(angles)
        y_components = np.sin(angles)

        # 对向量进行平滑处理
        pad_size = window_size // 2
        x_padded = np.pad(x_components, (pad_size, pad_size), 'edge')
        y_padded = np.pad(y_components, (pad_size, pad_size), 'edge')

        kernel = np.ones(window_size) / window_size
        x_smooth = np.convolve(x_padded, kernel, mode='valid')
        y_smooth = np.convolve(y_padded, kernel, mode='valid')

        # 转换回角度
        smoothed_angles = np.arctan2(y_smooth, x_smooth)

        return smoothed_angles

    def create_sector_points(self, center, radius, angle, facing_angle, num_points=20):
        """创建扇形区域的点集"""
        # 扇形的半角（弧度）
        half_angle = np.radians(angle) / 2

        # 计算扇形的起始和结束角度（弧度）
        start_angle = facing_angle - half_angle
        end_angle = facing_angle + half_angle

        # 创建扇形的点集
        points = [center]  # 扇形的中心点

        # 添加扇形弧上的点
        angle_step = (end_angle - start_angle) / num_points
        for i in range(num_points + 1):
            current_angle = start_angle + i * angle_step
            x = center[0] + radius * np.cos(current_angle)
            y = center[1] + radius * np.sin(current_angle)
            points.append((x, y))

        return np.array(points, dtype=np.int32)

    def update_trajectory(self, x, y, w_box, h_box, frame):
        """更新目标轨迹和朝向信息 - 改进数据记录完整性"""
        # 获取目标中心点的位置
        target_center = (x + w_box / 2, y + h_box / 2)
        position = np.array([target_center], dtype=np.float32).reshape(-1, 1, 2)

        # 将目标位置转换到初始帧的坐标系中
        position_in_initial = cv2.perspectiveTransform(position, self.cumulative_transform)
        position_in_initial = position_in_initial[0, 0]  # Shape: (2,)

        # 验证位置数据的有效性
        if np.isnan(position_in_initial).any() or np.isinf(position_in_initial).any():
            print(f"警告：位置数据无效，跳过此帧")
            return

        # 将轨迹点添加到列表
        self.trajectory.append(position_in_initial)

        # 记录当前帧的bounding box（在当前帧坐标系中）
        bbox_data = {
            'x': float(x),
            'y': float(y),
            'width': float(w_box),
            'height': float(h_box),
            'center': [float(target_center[0]), float(target_center[1])],
            'global_center': [float(position_in_initial[0]), float(position_in_initial[1])]
        }
        self.bounding_boxes.append(bbox_data)

        # 记录相机变换矩阵（深拷贝以避免引用问题）
        self.camera_transforms.append(self.cumulative_transform.copy())

        # 使用特征匹配估计朝向
        bbox = (x, y, w_box, h_box)
        expansion = self.config.get('feature_expansion', 20)

        direction_angle = None
        confidence = 1.0

        if self.config.get('use_feature_orientation', True):
            # 使用特征匹配方法
            direction_angle, vis_data = self.estimate_orientation_by_features(frame, bbox, expansion)

            # 提取置信度（如果可用）
            if isinstance(vis_data, dict):
                confidence = vis_data.get('inlier_ratio', 1.0)
                # 保存特征匹配的可视化数据用于绘制
                self.feature_vis_data = vis_data
            else:
                # 如果是矩形拟合方法返回的点集，保存为fitted_rectangle
                self.fitted_rectangle = vis_data
                self.feature_vis_data = None
        else:
            # 使用原始矩形拟合方法
            direction_angle, fitted_box = self.estimate_orientation_by_rectangle(frame, bbox, expansion)
            self.fitted_rectangle = fitted_box
            confidence = 1.0

        # 如果方向估计失败，则回退到运动方向
        if direction_angle is None:
            current_position = position_in_initial.copy()
            if self.prev_position is not None:
                # 计算移动向量和距离
                dx = current_position[0] - self.prev_position[0]
                dy = current_position[1] - self.prev_position[1]
                movement_distance = np.sqrt(dx * dx + dy * dy)

                # 只有当移动距离超过阈值时才更新方向
                if movement_distance > self.config['min_movement_threshold']:
                    direction_angle = np.arctan2(dy, dx)
                    confidence = min(movement_distance / 10.0, 1.0)
                elif len(self.direction_angles) > 0:
                    # 如果移动太小，使用前一个方向
                    direction_angle = self.direction_angles[-1]
                    confidence = 0.3
                else:
                    # 第一次且移动很小时，默认方向为0
                    direction_angle = 0
                    confidence = 0.1
            else:
                # 第一帧时没有朝向信息，使用初始朝向
                direction_angle = getattr(self, 'initial_orientation', 0)
                confidence = 0.1

        # 验证角度数据的有效性
        if np.isnan(direction_angle) or np.isinf(direction_angle):
            if len(self.direction_angles) > 0:
                direction_angle = self.direction_angles[-1]
            else:
                direction_angle = 0.0
            confidence = 0.1

        # 使用卡尔曼滤波平滑角度
        if self.config.get('use_kalman_filter', True):
            filtered_angle = self.update_angle_with_kalman(direction_angle, confidence)
        else:
            filtered_angle = direction_angle

        # 将方向角度添加到列表
        self.direction_angles.append(filtered_angle)

        # 更新前一帧位置
        self.prev_position = position_in_initial.copy()

        # 记录完整的帧数据
        frame_info = {
            'frame_index': len(self.frame_data),
            'robot_bbox': bbox_data,
            'robot_global_position': [float(position_in_initial[0]), float(position_in_initial[1])],
            'robot_orientation': float(filtered_angle),
            'camera_transform': self.cumulative_transform.tolist(),
            'confidence': float(confidence)
        }
        self.frame_data.append(frame_info)

        # 更新轨迹与方向的组合列表
        self.trajectory_with_direction = [(self.trajectory[i], self.direction_angles[i])
                                          for i in range(len(self.trajectory))]

    def calculate_camera_trajectory(self):
        """从相机变换矩阵计算相机在全局坐标系中的轨迹 - 修复版本"""
        camera_positions = []
        camera_orientations = []

        # 初始相机位置为原点
        initial_camera_pos = [0.0, 0.0]
        camera_positions.append(initial_camera_pos)
        camera_orientations.append(0.0)  # 初始朝向

        for i, transform in enumerate(self.camera_transforms):
            try:
                # 验证变换矩阵的有效性
                if transform is None or transform.shape != (3, 3):
                    print(f"警告：帧{i}的变换矩阵无效，使用上一帧数据")
                    if len(camera_positions) > 0:
                        camera_positions.append(camera_positions[-1])
                        camera_orientations.append(camera_orientations[-1])
                    else:
                        camera_positions.append([0.0, 0.0])
                        camera_orientations.append(0.0)
                    continue

                # 检查矩阵是否可逆
                det = np.linalg.det(transform)
                if abs(det) < 1e-6:
                    print(f"警告：帧{i}的变换矩阵接近奇异，使用上一帧数据")
                    if len(camera_positions) > 0:
                        camera_positions.append(camera_positions[-1])
                        camera_orientations.append(camera_orientations[-1])
                    else:
                        camera_positions.append([0.0, 0.0])
                        camera_orientations.append(0.0)
                    continue

                # 计算相机在全局坐标系中的位置
                # 相机变换矩阵的逆变换的平移部分就是相机位置
                inv_transform = np.linalg.inv(transform)
                camera_x = float(inv_transform[0, 2])
                camera_y = float(inv_transform[1, 2])

                # 验证坐标的有效性
                if np.isnan(camera_x) or np.isnan(camera_y) or np.isinf(camera_x) or np.isinf(camera_y):
                    print(f"警告：帧{i}的相机坐标无效，使用上一帧数据")
                    if len(camera_positions) > 0:
                        camera_positions.append(camera_positions[-1])
                        camera_orientations.append(camera_orientations[-1])
                    else:
                        camera_positions.append([0.0, 0.0])
                        camera_orientations.append(0.0)
                    continue

                camera_positions.append([camera_x, camera_y])

                # 从变换矩阵中提取旋转角度
                camera_angle = np.arctan2(inv_transform[1, 0], inv_transform[0, 0])

                # 验证角度的有效性
                if np.isnan(camera_angle) or np.isinf(camera_angle):
                    if len(camera_orientations) > 0:
                        camera_angle = camera_orientations[-1]
                    else:
                        camera_angle = 0.0

                camera_orientations.append(float(camera_angle))

            except Exception as e:
                # 如果计算失败，使用前一帧的值
                print(f"计算相机轨迹失败(帧{i}): {e}")
                if len(camera_positions) > 0:
                    camera_positions.append(camera_positions[-1])
                    camera_orientations.append(camera_orientations[-1])
                else:
                    camera_positions.append([0.0, 0.0])
                    camera_orientations.append(0.0)

        return camera_positions, camera_orientations

    def update_camera_motion(self, frame):
        """更新相机运动估计 - 使用选择的运动估计器"""
        # 获取当前跟踪目标的位置
        success, bbox = self.tracker.update(frame.copy())
        if not success:
            print("目标跟踪失败，无法更新相机运动估计")
            return

        x, y, w_box, h_box = [int(v) for v in bbox]

        # 创建排除目标的掩码
        exclusion_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # 计算排除区域（目标边界框向外扩展80像素）
        ex_x1 = max(0, x - 80)
        ex_y1 = max(0, y - 80)
        ex_x2 = min(frame.shape[1], x + w_box + 80)
        ex_y2 = min(frame.shape[0], y + h_box + 80)

        # 将目标区域及周围像素设为0（排除）
        exclusion_mask[ex_y1:ex_y2, ex_x1:ex_x2] = 0

        # 使用选择的运动估计器估计相机运动
        m_homo = self.motion_estimator.estimate(self.prev_frame, frame, exclusion_mask)

        # 更新前一帧
        self.prev_frame = frame.copy()

        # 更新累计变换矩阵 - 使用逆变换（从当前帧到初始帧）
        inv_m_homo = np.linalg.inv(m_homo)
        self.cumulative_transform = self.cumulative_transform @ inv_m_homo

        # 更新前一帧的灰度图像
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def draw_trajectory(self, frame):
        """在当前帧上绘制轨迹和方向信息，使用轨迹延迟参数"""
        # 对轨迹点进行平滑处理
        if len(self.trajectory) < 2:
            return

        # 使用高效数据结构获取轨迹点
        trajectory_array = self.trajectory.get_numpy_array()
        smoothing_size = self.config['smoothing_window_size']

        if len(trajectory_array) >= smoothing_size:
            # 使用边界值填充，减少边缘效应
            pad_size = smoothing_size // 2
            x_padded = np.pad(trajectory_array[:, 0], (pad_size, pad_size), 'edge')
            y_padded = np.pad(trajectory_array[:, 1], (pad_size, pad_size), 'edge')

            kernel = np.ones(smoothing_size) / smoothing_size
            x_smooth = np.convolve(x_padded, kernel, mode='valid')
            y_smooth = np.convolve(y_padded, kernel, mode='valid')

            smoothed_trajectory = np.vstack((x_smooth, y_smooth)).T
        else:
            smoothed_trajectory = trajectory_array

        # 将平滑后的轨迹点转换回当前帧坐标系
        smoothed_trajectory_points = smoothed_trajectory.reshape(-1, 1, 2).astype(np.float32)
        inv_transform = np.linalg.inv(self.cumulative_transform)

        # 使用GPU加速（如果可用）或CPU
        if self.config.get('use_gpu_acceleration', False) and hasattr(self, 'gpu_transform'):
            smoothed_trajectory_current = np.array(
                self.gpu_transform.transform_points_gpu(smoothed_trajectory.tolist(), inv_transform)
            ).reshape(-1, 1, 2)
        else:
            smoothed_trajectory_current = cv2.perspectiveTransform(smoothed_trajectory_points, inv_transform)

        # 创建一个只有轨迹线的二值掩码(白色=轨迹线，黑色=背景)
        binary_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # 如果启用了轨迹延迟参数
        trajectory_delay = self.config.get('trajectory_delay_distance', 0)

        # 应用轨迹延迟：只绘制满足延迟距离的轨迹段
        if trajectory_delay > 0 and len(smoothed_trajectory_current) > 1:
            for i in range(1, len(smoothed_trajectory_current)):
                # 计算当前点到最新点的距离
                latest_pt = smoothed_trajectory_current[-1][0]
                current_pt = smoothed_trajectory_current[i][0]
                prev_pt = smoothed_trajectory_current[i - 1][0]

                # 计算当前点与最新点的距离
                dx = current_pt[0] - latest_pt[0]
                dy = current_pt[1] - latest_pt[1]
                distance = np.sqrt(dx * dx + dy * dy)

                # 只有当距离大于设定的延迟距离时才绘制轨迹
                if distance >= trajectory_delay:
                    pt1 = (int(prev_pt[0]), int(prev_pt[1]))
                    pt2 = (int(current_pt[0]), int(current_pt[1]))
                    cv2.line(binary_mask, pt1, pt2, 255, thickness=self.config['trajectory_thickness'])
        else:
            # 不需要延迟，正常绘制全部轨迹
            for i in range(1, len(smoothed_trajectory_current)):
                pt1 = (int(smoothed_trajectory_current[i - 1][0][0]), int(smoothed_trajectory_current[i - 1][0][1]))
                pt2 = (int(smoothed_trajectory_current[i][0][0]), int(smoothed_trajectory_current[i][0][1]))
                cv2.line(binary_mask, pt1, pt2, 255, thickness=self.config['trajectory_thickness'])

        # 创建与frame相同大小的彩色轨迹图层
        trajectory_color_mask = np.zeros_like(frame)
        # 填充整个图层为轨迹颜色
        trajectory_color_mask[:] = self.config['trajectory_color']

        # 获取透明度值（0-1之间）
        trajectory_alpha = min(max(self.config.get('trajectory_alpha', 1.0), 0), 1)

        # 使用二值掩码进行真正的alpha混合
        # 只在轨迹线的位置进行原图和颜色的混合
        overlay = np.where(
            binary_mask[:, :, np.newaxis] > 0,  # 在轨迹线的位置
            # 在轨迹位置混合原图和轨迹颜色
            cv2.addWeighted(
                frame,
                1.0 - trajectory_alpha,  # 原图权重 = 1-透明度
                trajectory_color_mask,
                trajectory_alpha,  # 轨迹颜色权重 = 透明度
                0
            ),
            frame  # 非轨迹位置保持原图不变
        )

        # 绘制拟合的矩形（如果存在且启用显示）
        if self.config.get('show_fitted_rectangle', True) and self.fitted_rectangle is not None:
            cv2.drawContours(overlay, [self.fitted_rectangle], 0, (0, 255, 255), 2)

        # 只在最新位置绘制朝向箭头和扇形区域
        if len(self.trajectory_with_direction) > 0:
            # 获取最新位置
            latest_position = smoothed_trajectory_current[-1][0]
            latest_pt = (int(latest_position[0]), int(latest_position[1]))

            # 获取最新朝向角度
            _, latest_angle = self.trajectory_with_direction[-1]

            # # 如果有特征匹配的可视化数据，直接使用
            # if hasattr(self, 'feature_vis_data') and self.feature_vis_data:
            #     # 使用特征匹配返回的点
            #     latest_pt = self.feature_vis_data['center']

            # 计算箭头终点
            arrow_length = self.config['direction_arrow_length']
            arrow_end_x = latest_pt[0] + int(arrow_length * np.cos(latest_angle))
            arrow_end_y = latest_pt[1] + int(arrow_length * np.sin(latest_angle))
            arrow_end = (arrow_end_x, arrow_end_y)

            # 绘制箭头 - 使用相同的透明度方法
            arrow_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.arrowedLine(arrow_mask, latest_pt, arrow_end, 255, 2, tipLength=0.3)

            # 获取箭头透明度
            arrow_alpha = min(max(self.config.get('arrow_alpha', trajectory_alpha), 0), 1)

            # 创建箭头颜色层
            arrow_color_mask = np.zeros_like(frame)
            arrow_color_mask[:] = self.config['direction_arrow_color']

            # 应用箭头透明度
            overlay = np.where(
                arrow_mask[:, :, np.newaxis] > 0,
                cv2.addWeighted(
                    overlay,
                    1.0 - arrow_alpha,
                    arrow_color_mask,
                    arrow_alpha,
                    0
                ),
                overlay
            )

            # 创建扇形区域的点集
            sector_points = self.create_sector_points(
                latest_pt,
                self.config['sector_radius'],
                self.config['sector_angle'],
                latest_angle
            )

            # 创建扇形掩码
            sector_mask = np.zeros_like(frame)
            cv2.fillPoly(sector_mask, [sector_points], self.config['sector_color'])

            # 应用扇形透明度
            sector_alpha = self.config['sector_alpha']
            overlay = cv2.addWeighted(overlay, 1, sector_mask, sector_alpha, 0)

        # 将结果复制到原始帧
        frame[:] = overlay

    def process_frame(self):
        """处理单帧视频并更新跟踪和显示"""
        ret, frame = self.cap.read()
        if not ret:
            return False  # 视频结束

        try:
            # 开始帧处理计时（用于FPS计算）
            self.original_frame = frame.copy()
            display_frame = frame.copy()
            start_time = time.time()

            # 更新目标追踪器
            success, bbox = self.tracker.update(self.original_frame)
            if not success:
                print("目标丢失，停止追踪")
                return False

            x, y, w_box, h_box = [int(v) for v in bbox]
            # 更新相机运动估计
            self.update_camera_motion(self.original_frame)

            # 更新目标轨迹 - 传递当前帧用于矩形拟合
            self.update_trajectory(x, y, w_box, h_box, self.original_frame)

            cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            # 绘制轨迹和方向
            self.draw_trajectory(display_frame)

            # 保存当前帧用于标注
            self.current_frame = display_frame.copy()

            # 渲染标注
            self.render_annotations(display_frame)
            self.display_frame = display_frame.copy()

            # 添加水印（如果需要）
            if self.config['add_watermark']:
                frame = cv2.addWeighted(display_frame, 1, self.watermark, 0.3, 0)

            # 显示结果
            cv2.imshow(self.main_window_name, display_frame)

            # 如果需要记录视频，写入帧
            if self.config['record_video']:
                self.out.write(display_frame)

            # 计算并存储帧处理时间
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)

            return True  # 成功处理帧

        except Exception as e:
            print(f"处理帧时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def render_annotations_refined(self, frame):
        """渲染refined模式的标注 - 只显示已探索的像素"""
        if not self.refine_mode or not self.exploration_manager:
            # 如果未启用refine模式，回退到原始渲染
            self.render_annotations(frame)
            return

        # 获取逆变换矩阵（从全局坐标系到当前帧坐标系）
        inv_transform = np.linalg.inv(self.cumulative_transform)

        # 渲染refined多边形
        for i, global_polygon in enumerate(self.polygons):
            if i >= len(self.annotation_indices['polygons']):
                continue

            annotation_idx = self.annotation_indices['polygons'][i]

            # 获取已探索的像素掩码
            explored_mask = self.exploration_manager.get_explored_annotation_pixels('polygon', annotation_idx)
            if explored_mask is None or np.sum(explored_mask) == 0:
                continue  # 没有探索到的像素，跳过

            # 将全局掩码转换到当前帧坐标系
            local_explored_mask = self._transform_mask_to_current_frame(explored_mask, inv_transform, frame.shape)

            if local_explored_mask is not None:
                # 应用探索掩码渲染多边形
                self._render_masked_annotation(frame, local_explored_mask, 'polygon')

        # 渲染refined圆形
        for i, (global_center, global_radius) in enumerate(self.circles):
            if i >= len(self.annotation_indices['circles']):
                continue

            annotation_idx = self.annotation_indices['circles'][i]

            # 获取已探索的像素掩码
            explored_mask = self.exploration_manager.get_explored_annotation_pixels('circle', annotation_idx)
            if explored_mask is None or np.sum(explored_mask) == 0:
                continue  # 没有探索到的像素，跳过

            # 将全局掩码转换到当前帧坐标系
            local_explored_mask = self._transform_mask_to_current_frame(explored_mask, inv_transform, frame.shape)

            if local_explored_mask is not None:
                # 应用探索掩码渲染圆形
                self._render_masked_annotation(frame, local_explored_mask, 'circle')

    def _transform_mask_to_current_frame(self, global_mask, inv_transform, frame_shape):
        """将全局掩码转换到当前帧坐标系

        Args:
            global_mask (ndarray): 全局坐标系中的掩码
            inv_transform (ndarray): 逆变换矩阵
            frame_shape (tuple): 当前帧的形状

        Returns:
            ndarray: 当前帧坐标系中的掩码
        """
        try:
            # 使用warpPerspective进行掩码变换
            local_mask = cv2.warpPerspective(
                global_mask,
                inv_transform,
                (frame_shape[1], frame_shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            return local_mask
        except Exception as e:
            print(f"掩码变换失败: {e}")
            return None

    def _render_masked_annotation(self, frame, mask, annotation_type):
        """渲染被掩码限制的标注

        Args:
            frame (ndarray): 目标帧
            mask (ndarray): 像素掩码
            annotation_type (str): 标注类型 'polygon' 或 'circle'
        """
        # 根据标注类型选择颜色和透明度
        if annotation_type == 'polygon':
            fill_color = self.config['polygon_fill_color']
            border_color = self.config['polygon_border_color']
            border_thickness = self.config['polygon_border_thickness']
            alpha = self.config['polygon_alpha']
        else:  # circle
            fill_color = self.config['circle_fill_color']
            border_color = self.config['circle_border_color']
            border_thickness = self.config['circle_border_thickness']
            alpha = self.config['circle_alpha']

        # 创建颜色掩码
        color_mask = np.zeros_like(frame)
        color_mask[mask > 0] = fill_color

        # 应用透明度
        cv2.addWeighted(frame, 1, color_mask, alpha, 0, frame)

        # 添加边框（基于掩码的轮廓）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, border_color, border_thickness)

    def run(self):
        """运行主循环"""
        # 显示初始操作提示
        print("\n=== 开始运行 ===")
        print("按空格暂停视频，然后按'p'绘制多边形或按'o'绘制圆形")
        print("绘制多边形时：点击添加顶点，连线实时显示，按Enter保存，按Backspace撤销点")
        print("绘制圆形时：拖动鼠标绘制圆形，释放确认，按Enter保存，按Backspace撤销")

        while True:
            # 如果是播放状态，处理帧
            if self.current_state == State.PLAYING:
                if not self.process_frame():
                    break

            # 处理键盘输入
            key = cv2.waitKey(1 if self.current_state == State.PLAYING else 0) & 0xFF
            if not self.handle_key(key):
                break

        # 释放资源
        self.cap.release()
        if self.config['record_video']:
            self.out.release()
        cv2.destroyAllWindows()

        # 自动导出追踪数据
        print("\n=== 视频处理完成，正在导出追踪数据 ===")
        export_path = self.config.get('tracking_data_export_path', 'tracking_data_5_29_good.json')
        self.export_tracking_data(export_path)
        print("=== 导出完成！现在可以运行refine_real_world.py进行refined渲染 ===")

    def convert_numpy_types(self, obj):
        """递归转换NumPy数据类型为Python原生类型 - 增强版"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj


def main():
    """主函数，设置配置并启动应用"""
    # 集中配置参数
    config = {
        # 基本配置
        'video_path': 'real_world_video.mp4',
        'output_video_path': 'annotated_trajectory.mp4',
        'record_video': True,
        'add_watermark': False,
        'visualize_features': False,
        'watermark_text': 'TIANLIANG',
        'show_fps': True,  # 显示FPS
        'debug_orientation': False,  # 启用朝向估计调试窗口
        'debug_feature_matching': False,  # 启用特征匹配调试窗口
        'auto_continue_on_poor_quality': False,  # 即使质量差也继续运行，不请求人工校准

        # 模板存储和加载相关
        'template_save_path': '/home/lzh/NewCppRL/utils/templates/',  # 模板存储路径
        'load_saved_templates': True,  # 是否加载已保存的模板
        'orientation_jump_threshold': 30,  # 角度跳变阈值(度)
        'min_match_quality': 0.39,  # 最低匹配质量阈值

        # 特征提取和跟踪相关
        'use_cnn_features': True,  # 使用CNN特征提取器
        'cnn_model': 'mobilenet_v2',  # 可选: 'mobilenet_v2', 'resnet18'
        'tracker_type': 'CSRT',  # 可选: 'CSRT', 'KCF', 'DEEPSORT' 等

        # 运动估计相关
        'use_depth_motion': True,  # 使用深度辅助的运动估计器
        'depth_model_type': 'MiDaS_small',  # 可选: 'MiDaS_small', 'DPT_Large', 'DPT_Hybrid'

        # 性能优化
        'use_gpu_acceleration': True,  # 使用GPU加速坐标变换
        'max_trajectory_points': 5000,  # 存储的最大轨迹点数量

        # 特征匹配朝向估计相关
        'skip_initial_orientation_setup': False,
        'use_feature_orientation': True,  # 使用特征匹配朝向估计
        'template_expansion': 120,  # 模板边界框扩展像素数
        'feature_expansion': 120,  # 匹配时边界框扩展像素数

        # SIFT参数
        # 'sift_features': 1500,  # 从500增加到1500
        # 'sift_contrast': 0.02,  # 从0.04降低到0.02
        # 'sift_edge': 7,  # 从10降低到7

        'sift_features': 500,  # SIFT特征点数量
        'sift_contrast': 0.04,  # SIFT对比度阈值
        'sift_edge': 10,  # SIFT边缘阈值

        # 匹配参数
        'match_ratio': 0.7,  # 特征匹配比率测试阈值
        'ransac_threshold': 3.0,  # RANSAC重投影阈值
        'ransac_confidence': 0.99,  # RANSAC置信度
        'ransac_iterations': 2000,  # RANSAC最大迭代次数

        # 卡尔曼滤波相关
        'use_kalman_filter': True,  # 使用卡尔曼滤波平滑角度

        # 原始矩形拟合朝向估计相关（用作备选）
        'rect_expansion': 15,  # 边界框扩展像素数
        'show_fitted_rectangle': True,  # 是否显示拟合的矩形

        # 追踪参数
        'max_corners': 100,
        'quality_level': 0.01,
        'min_distance': 15,
        'block_size': 3,
        'min_movement_threshold': 0.1,

        # 轨迹参数
        'smoothing_window_size': 5,
        'direction_smoothing_window_size': 5,
        'direction_arrow_length': 30,
        'trajectory_delay_distance': 0,  # 轨迹延迟距离（像素）

        # 轨迹颜色
        'trajectory_alpha': 1,  # 0.75,
        'trajectory_thickness': 4,  # 20,
        'trajectory_color': (0, 0, 220),  # 红色
        'direction_arrow_color': (0, 255, 0),  # 绿色
        'arrow_alpha': 1,

        # 扇形参数
        'sector_angle': 75,
        'sector_radius': 650,
        'sector_color': (229, 153, 51),  # 蓝色
        'sector_alpha': 0.3,

        # 多边形标注参数（障碍物/石头）
        'polygon_point_color': (40, 40, 180),  # 紫色点
        'polygon_line_color': (40, 40, 180),  # 紫色线
        'polygon_fill_color': (0, 0, 220),  # 红色填充
        'polygon_border_color': (0, 0, 160),  # 深红色边框
        'polygon_border_thickness': 2,  # 边框厚度
        'polygon_alpha': 0.6,  # 透明度

        # 圆形标注参数（杂草）
        'circle_line_color': (0, 255, 0),  # 绿色
        'circle_fill_color': (0, 220, 0),  # 绿色填充
        'circle_border_color': (0, 150, 0),  # 深绿色边框
        'circle_border_thickness': 2,  # 边框厚度
        'circle_alpha': 0.5,  # 透明度
    }

    # 检查GPU可用性并调整配置
    if not GPU_AVAILABLE:
        config['use_gpu_acceleration'] = False
        print("GPU加速不可用，已禁用GPU加速选项")

    # 检查PyTorch GPU可用性并调整配置
    if not TORCH_GPU_AVAILABLE:
        print("PyTorch GPU加速不可用，这可能会减慢CNN特征提取和深度估计的速度")

    # 检查深度学习库的可用性，如果不可用则回退
    try:
        import torch
        import torchvision
    except ImportError:
        config['use_cnn_features'] = False
        config['use_depth_motion'] = False
        print("缺少PyTorch/torchvision，已禁用CNN特征和深度运动估计")

    # 检查DeepSORT的可用性
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError:
        if config['tracker_type'] == 'DEEPSORT':
            config['tracker_type'] = 'CSRT'
            print("缺少deep_sort_realtime库，已回退到CSRT跟踪器")
            print("要安装DeepSORT，请运行: pip install deep-sort-realtime")

    try:
        # 创建并运行应用
        app = VideoTracker(config)
        app.run()
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()