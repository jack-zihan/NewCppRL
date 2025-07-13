#!/usr/bin/env python3
"""
Feature Visualizer - 特征可视化工具（修复版）
基于PyQt5的专业特征可视化界面
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'

# Try to import cpu_apf_bool, provide fallback if not available
try:
    from cpu_apf import cpu_apf_bool
except ImportError:
    print("Warning: cpu_apf_bool not found. Using fallback implementation.")

    raise Exception("cpu_apf_bool is required but not found. Please install the cpu_apf package.")
    def cpu_apf_bool(map_apf):
        # Simple fallback implementation using distance transform
        from scipy import ndimage
        if map_apf.sum() == 0:
            return map_apf, True
        # Convert to binary
        binary_map = map_apf.astype(bool)
        # Distance transform
        distances = ndimage.distance_transform_edt(~binary_map)
        return distances, False

# Import total_variation_mat
try:
    from envs.utils import total_variation_mat
except ImportError:
    # raise Exception("total_variation_mat is required but not found. Please ensure it is available in the envs.utils module.")
    # def total_variation_mat(image: np.ndarray) -> np.ndarray:
    #     """Fallback implementation of total variation."""
    #     mask_tv_cols = np.abs(image[1:, :] - image[:-1, :]) > 0
    #     mask_tv_cols = np.pad(mask_tv_cols, pad_width=[[0, 1], [0, 0]], mode='constant')
    #     mask_tv_rows = np.abs(image[:, 1:] - image[:, :-1]) > 0
    #     mask_tv_rows = np.pad(mask_tv_rows, pad_width=[[0, 0], [0, 1]], mode='constant')
    #     mask_tv = np.logical_or(mask_tv_rows, mask_tv_cols)
    #     return mask_tv
    def total_variation_mat(mat):
        mat = np.asarray(mat)

        tv = np.zeros(mat.shape, dtype=bool)

        tv[1:, :] |= (mat[1:, :] != mat[:-1, :])
        tv[:-1, :] |= (mat[1:, :] != mat[:-1, :])

        tv[:, 1:] |= (mat[:, 1:] != mat[:, :-1])
        tv[:, :-1] |= (mat[:, 1:] != mat[:, :-1])

        return tv

class VisualizationCanvas(FigureCanvas):
    """自定义的matplotlib画布"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.patch.set_facecolor('white')

        # 设置大小策略
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class FeatureCheckBox(QWidget):
    """自定义的特征复选框组件"""

    stateChanged = pyqtSignal(str, str, bool)  # feature_type, subtype, checked

    def __init__(self, feature_type: str, subtypes: List[str], color: Tuple[float, float, float]):
        super().__init__()
        self.feature_type = feature_type
        self.subtypes = subtypes
        self.color = color
        self.checkboxes = {}

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        # Feature label with color
        label = QLabel(f"{self.feature_type.capitalize()}:")
        label.setStyleSheet(f"""
            QLabel {{
                color: rgb({int(self.color[0] * 255)}, {int(self.color[1] * 255)}, {int(self.color[2] * 255)});
                font-weight: bold;
                font-size: 12px;
                min-width: 80px;
            }}
        """)
        layout.addWidget(label)

        # Checkboxes for subtypes
        for subtype in ['full', 'raw', 'apf']:
            if subtype in self.subtypes:
                checkbox = QCheckBox(subtype)
                checkbox.setStyleSheet("""
                    QCheckBox {
                        font-size: 11px;
                        spacing: 5px;
                    }
                    QCheckBox::indicator {
                        width: 13px;
                        height: 13px;
                    }
                """)
                checkbox.stateChanged.connect(
                    lambda state, st=subtype: self.on_state_changed(st, state)
                )
                self.checkboxes[subtype] = checkbox
                layout.addWidget(checkbox)
            else:
                # Add spacer for missing subtypes
                spacer = QLabel("")
                spacer.setFixedWidth(50)
                layout.addWidget(spacer)

        layout.addStretch()
        self.setLayout(layout)

    def on_state_changed(self, subtype: str, state: int):
        self.stateChanged.emit(self.feature_type, subtype, state == Qt.Checked)

    def set_states_without_signal(self, states: Dict[str, bool]):
        """设置状态但不触发信号"""
        for st, checked in states.items():
            if st in self.checkboxes:
                self.checkboxes[st].blockSignals(True)
                self.checkboxes[st].setChecked(checked)
                self.checkboxes[st].blockSignals(False)

    def get_states(self) -> Dict[str, bool]:
        return {st: cb.isChecked() for st, cb in self.checkboxes.items()}

    def set_states(self, states: Dict[str, bool]):
        for st, checked in states.items():
            if st in self.checkboxes:
                self.checkboxes[st].setChecked(checked)

    def set_states_without_signal(self, states: Dict[str, bool]):
        """设置状态但不触发信号"""
        for st, checked in states.items():
            if st in self.checkboxes:
                self.checkboxes[st].blockSignals(True)
                self.checkboxes[st].setChecked(checked)
                self.checkboxes[st].blockSignals(False)


class APFParameterWidget(QWidget):
    """APF参数控制组件"""

    valueChanged = pyqtSignal(str, int)  # feature_type, value

    def __init__(self, feature_type: str, color: Tuple[float, float, float],
                 default_value: int = 30):
        super().__init__()
        self.feature_type = feature_type
        self.color = color
        self.value = default_value

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Label
        label = QLabel(f"{self.feature_type}:")
        label.setStyleSheet(f"""
            QLabel {{
                color: rgb({int(self.color[0] * 255)}, {int(self.color[1] * 255)}, {int(self.color[2] * 255)});
                font-size: 11px;
                min-width: 60px;
            }}
        """)
        layout.addWidget(label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(self.value)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #E0E0E0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078D4;
                border: 1px solid #0078D4;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        # SpinBox
        self.spinbox = QSpinBox()
        self.spinbox.setRange(1, 100)
        self.spinbox.setValue(self.value)
        self.spinbox.setFixedWidth(60)
        self.spinbox.setStyleSheet("""
            QSpinBox {
                padding: 2px;
                font-size: 11px;
            }
        """)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)
        layout.addWidget(self.spinbox)

        self.setLayout(layout)

    def on_slider_changed(self, value: int):
        self.value = value
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(self.feature_type, value)

    def on_spinbox_changed(self, value: int):
        self.value = value
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(self.feature_type, value)

    def get_value(self) -> int:
        return self.value

    def set_value(self, value: int):
        self.value = value
        self.slider.setValue(value)
        self.spinbox.setValue(value)


class FeatureVisualizerGUI(QMainWindow):
    """Feature可视化主窗口"""

    # Feature colors matching RL environment
    COLORS = {
        'frontier': (112 / 255, 173 / 255, 7 / 255),  # Green
        'weed': (255 / 255, 0 / 255, 0 / 255),  # Red
        'obstacle': (30 / 255, 75 / 255, 130 / 255),  # Blue
        'trajectory': (255 / 255, 38 / 255, 255 / 255),  # Magenta
    }

    # Feature types
    FEATURE_TYPES = ['frontier', 'weed', 'obstacle', 'trajectory']
    FEATURE_SUBTYPES = {
        'frontier': ['full', 'raw', 'apf'],
        'weed': ['full', 'raw', 'apf'],
        'obstacle': ['raw', 'apf'],
        'trajectory': ['raw', 'apf'],
    }

    # Default APF parameters
    DEFAULT_APF_PARAMS = {
        'frontier': 30,
        'weed': 40,
        'obstacle': 10,
        'trajectory': 4,
    }

    DEFAULT_APF_EPS = {
        'frontier': None,
        'weed': 1e-2,
        'obstacle': None,
        'trajectory': None,
    }

    def __init__(self):
        super().__init__()

        # Data
        self.feature_dir = None
        self.frames = []
        self.current_frame_idx = 0
        self.current_data = {}
        self.current_metadata = {}

        # UI state
        self.enabled_features = {}
        for feat in self.FEATURE_TYPES:
            self.enabled_features[feat] = {}
            for subtype in self.FEATURE_SUBTYPES[feat]:
                self.enabled_features[feat][subtype] = False

        self.enabled_features['mist'] = False
        self.enabled_features['rendered_map'] = False

        # 设置一些默认勾选的选项，方便用户快速开始
        self.enabled_features['frontier']['raw'] = True
        self.enabled_features['frontier']['apf'] = True
        self.enabled_features['weed']['raw'] = True
        self.enabled_features['obstacle']['raw'] = True
        self.enabled_features['obstacle']['apf'] = True

        # Visualization mode
        self.aggregate_mode = False

        # APF parameters
        self.apf_params = self.DEFAULT_APF_PARAMS.copy()
        self.apf_pad_obstacle = True

        # UI components
        self.feature_checkboxes = {}
        self.apf_widgets = {}

        self.init_ui()
        self.setup_styles()

        # 应用默认勾选状态到UI
        QTimer.singleShot(100, self.apply_default_selections)

        # 标记初始化完成
        QTimer.singleShot(200, lambda: setattr(self, 'initialization_complete', True))

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('Feature Visualizer - 特征可视化工具')
        self.setGeometry(100, 100, 1600, 900)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # 左侧控制面板
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 0)

        # 右侧可视化区域
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 1)

        # 状态栏
        self.statusBar().showMessage('准备就绪 | 提示：使用左右箭头键快速切换帧')

        # 设置窗口图标（如果有的话）
        # self.setWindowIcon(QIcon('icon.png'))

    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(450)
        panel.setStyleSheet("""
            QWidget {
                background-color: #F8F9FA;
                border: 1px solid #DEE2E6;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 标题
        title = QLabel("控制面板")
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2E2E2E;
                border: none;
                padding: 5px 0;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #DEE2E6;")
        layout.addWidget(line)

        # 1. 文件选择
        file_group = self.create_file_selection_group()
        layout.addWidget(file_group)

        # 2. 帧选择
        frame_group = self.create_frame_selection_group()
        layout.addWidget(frame_group)

        # 3. 可视化模式
        mode_group = self.create_visualization_mode_group()
        layout.addWidget(mode_group)

        # 4. 特征选择
        feature_group = self.create_feature_selection_group()
        layout.addWidget(feature_group)

        # 5. APF参数
        apf_group = self.create_apf_parameters_group()
        layout.addWidget(apf_group)

        # 6. 操作按钮
        button_group = self.create_action_buttons_group()
        layout.addWidget(button_group)

        layout.addStretch()

        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #F0F0F0;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #CCCCCC;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #999999;
            }
        """)

        return scroll

    def create_file_selection_group(self) -> QGroupBox:
        """创建文件选择组"""
        group = QGroupBox("数据目录")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout()

        # 路径输入和浏览按钮
        path_layout = QHBoxLayout()

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("请选择或输入Feature数据目录...")
        self.path_input.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                font-size: 12px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }
            QLineEdit:focus {
                border-color: #0078D4;
            }
        """)
        path_layout.addWidget(self.path_input)

        browse_btn = QPushButton("浏览...")
        browse_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 15px;
                font-size: 12px;
                background-color: #0078D4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
        """)
        browse_btn.clicked.connect(self.browse_directory)
        path_layout.addWidget(browse_btn)

        layout.addLayout(path_layout)

        # 加载按钮
        load_btn = QPushButton("加载数据")
        load_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-size: 13px;
                background-color: #28A745;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1E7E34;
            }
        """)
        load_btn.clicked.connect(self.load_feature_directory)
        layout.addWidget(load_btn)

        group.setLayout(layout)
        return group

    def create_frame_selection_group(self) -> QGroupBox:
        """创建帧选择组"""
        group = QGroupBox("帧选择")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout()

        # 帧信息
        self.frame_info_label = QLabel("未加载数据")
        self.frame_info_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666666;
                padding: 5px;
            }
        """)
        layout.addWidget(self.frame_info_label)

        # 帧滑块
        slider_layout = QHBoxLayout()

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #E0E0E0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078D4;
                border: 1px solid #0078D4;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #106EBE;
                border-color: #106EBE;
            }
        """)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        slider_layout.addWidget(self.frame_slider)

        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setRange(0, 0)
        self.frame_spinbox.setFixedWidth(80)
        self.frame_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 4px;
                font-size: 12px;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }
        """)
        self.frame_spinbox.valueChanged.connect(self.on_frame_spinbox_changed)
        slider_layout.addWidget(self.frame_spinbox)

        layout.addLayout(slider_layout)

        # 快速跳转按钮
        nav_layout = QHBoxLayout()

        first_btn = QPushButton("⏮ 第一帧")
        first_btn.clicked.connect(lambda: self.jump_to_frame(0))
        nav_layout.addWidget(first_btn)

        prev_btn = QPushButton("◀ 上一帧")
        prev_btn.clicked.connect(lambda: self.jump_to_frame(self.current_frame_idx - 1))
        nav_layout.addWidget(prev_btn)

        next_btn = QPushButton("下一帧 ▶")
        next_btn.clicked.connect(lambda: self.jump_to_frame(self.current_frame_idx + 1))
        nav_layout.addWidget(next_btn)

        last_btn = QPushButton("最后帧 ⏭")
        last_btn.clicked.connect(lambda: self.jump_to_frame(len(self.frames) - 1))
        nav_layout.addWidget(last_btn)

        for btn in [first_btn, prev_btn, next_btn, last_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    padding: 4px 8px;
                    font-size: 11px;
                    background-color: #6C757D;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #5A6268;
                }
                QPushButton:pressed {
                    background-color: #545B62;
                }
            """)

        layout.addLayout(nav_layout)

        group.setLayout(layout)
        return group

    def create_visualization_mode_group(self) -> QGroupBox:
        """创建可视化模式组"""
        group = QGroupBox("可视化模式")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QHBoxLayout()

        self.separated_radio = QRadioButton("分离模式")
        self.separated_radio.setChecked(True)
        self.separated_radio.setStyleSheet("""
            QRadioButton {
                font-size: 12px;
                spacing: 5px;
            }
        """)
        layout.addWidget(self.separated_radio)

        self.aggregated_radio = QRadioButton("聚合模式")
        self.aggregated_radio.setStyleSheet("""
            QRadioButton {
                font-size: 12px;
                spacing: 5px;
            }
        """)
        layout.addWidget(self.aggregated_radio)

        # 创建按钮组
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.separated_radio, 0)
        self.mode_group.addButton(self.aggregated_radio, 1)
        self.mode_group.buttonClicked.connect(self.on_mode_changed)

        layout.addStretch()

        group.setLayout(layout)
        return group

    def create_feature_selection_group(self) -> QGroupBox:
        """创建特征选择组"""
        group = QGroupBox("特征选择")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)

        # 添加每个特征类型的复选框
        for feat_type in self.FEATURE_TYPES:
            checkbox_widget = FeatureCheckBox(
                feat_type,
                self.FEATURE_SUBTYPES[feat_type],
                self.COLORS[feat_type]
            )
            checkbox_widget.stateChanged.connect(self.on_feature_toggled)
            self.feature_checkboxes[feat_type] = checkbox_widget
            layout.addWidget(checkbox_widget)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #E0E0E0;")
        layout.addWidget(line)

        # 特殊特征
        special_layout = QHBoxLayout()

        self.mist_checkbox = QCheckBox("Mist")
        self.mist_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                spacing: 5px;
            }
        """)
        self.mist_checkbox.stateChanged.connect(
            lambda state: self.on_special_feature_toggled('mist', state == Qt.Checked)
        )
        special_layout.addWidget(self.mist_checkbox)

        self.rendered_checkbox = QCheckBox("Rendered Map")
        self.rendered_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                spacing: 5px;
            }
        """)
        self.rendered_checkbox.stateChanged.connect(
            lambda state: self.on_special_feature_toggled('rendered_map', state == Qt.Checked)
        )
        special_layout.addWidget(self.rendered_checkbox)

        special_layout.addStretch()
        layout.addLayout(special_layout)

        group.setLayout(layout)
        return group

    def create_apf_parameters_group(self) -> QGroupBox:
        """创建APF参数组"""
        group = QGroupBox("APF参数设置")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #E0E0E0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)

        # 添加每个特征的APF参数控制
        for feat_type in self.FEATURE_TYPES:
            apf_widget = APFParameterWidget(
                feat_type,
                self.COLORS[feat_type],
                self.DEFAULT_APF_PARAMS[feat_type]
            )
            apf_widget.valueChanged.connect(self.on_apf_value_changed)
            self.apf_widgets[feat_type] = apf_widget
            layout.addWidget(apf_widget)

        # Obstacle padding选项
        self.obstacle_padding_checkbox = QCheckBox("Obstacle Padding")
        self.obstacle_padding_checkbox.setChecked(True)
        self.obstacle_padding_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                margin-top: 10px;
            }
        """)
        self.obstacle_padding_checkbox.stateChanged.connect(
            lambda state: setattr(self, 'apf_pad_obstacle', state == Qt.Checked)
        )
        layout.addWidget(self.obstacle_padding_checkbox)

        # 重新计算按钮
        self.recalc_btn = QPushButton("重新计算 APF")
        self.recalc_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-size: 12px;
                background-color: #FFC107;
                color: #212529;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #E0A800;
            }
            QPushButton:pressed {
                background-color: #D39E00;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        self.recalc_btn.clicked.connect(self.recalculate_apf)
        self.recalc_btn.setEnabled(False)  # 初始状态禁用
        layout.addWidget(self.recalc_btn)

        group.setLayout(layout)
        return group

    def create_action_buttons_group(self) -> QWidget:
        """创建操作按钮组"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # 刷新视图按钮
        refresh_btn = QPushButton("刷新视图")
        refresh_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 13px;
                background-color: #17A2B8;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
            QPushButton:pressed {
                background-color: #117A8B;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_visualization)
        layout.addWidget(refresh_btn)

        # 导出图片按钮
        export_btn = QPushButton("导出图片")
        export_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 13px;
                background-color: #6F42C1;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A32A3;
            }
            QPushButton:pressed {
                background-color: #4E2A91;
            }
        """)
        export_btn.clicked.connect(self.export_figure)
        layout.addWidget(export_btn)

        # 添加一些说明文字
        info_label = QLabel("提示：支持导出PNG/PDF/SVG格式")
        info_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #666666;
                font-style: italic;
                margin-top: 5px;
            }
        """)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        return widget

    def create_visualization_panel(self) -> QWidget:
        """创建可视化面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建matplotlib画布
        self.canvas = VisualizationCanvas(self, width=12, height=8, dpi=100)
        self.canvas.setStyleSheet("""
            background-color: white;
            border: 1px solid #DEE2E6;
            border-radius: 8px;
        """)

        # 创建工具栏
        toolbar_widget = QWidget()
        toolbar_widget.setMaximumHeight(40)
        toolbar_widget.setStyleSheet("""
            QWidget {
                background-color: #F8F9FA;
                border: 1px solid #DEE2E6;
                border-radius: 4px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)

        # 元数据标签
        self.metadata_label = QLabel("准备就绪")
        self.metadata_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #495057;
                padding: 5px;
            }
        """)
        toolbar_layout.addWidget(self.metadata_label)
        toolbar_layout.addStretch()

        # 缩放控制
        zoom_label = QLabel("缩放:")
        zoom_label.setStyleSheet("font-size: 12px; color: #495057;")
        toolbar_layout.addWidget(zoom_label)

        zoom_in_btn = QPushButton("放大")
        zoom_in_btn.setFixedSize(60, 28)
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar_layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("缩小")
        zoom_out_btn.setFixedSize(60, 28)
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar_layout.addWidget(zoom_out_btn)

        zoom_reset_btn = QPushButton("重置")
        zoom_reset_btn.setFixedSize(60, 28)
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        toolbar_layout.addWidget(zoom_reset_btn)

        for btn in [zoom_in_btn, zoom_out_btn, zoom_reset_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 11px;
                    background-color: #6C757D;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #5A6268;
                }
                QPushButton:pressed {
                    background-color: #545B62;
                }
            """)

        layout.addWidget(toolbar_widget)
        layout.addWidget(self.canvas)

        return panel

    def setup_styles(self):
        """设置全局样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QMessageBox {
                background-color: white;
            }
            QToolTip {
                background-color: #FFFBDD;
                border: 1px solid #E0D268;
                padding: 5px;
                font-size: 11px;
            }
        """)

    def apply_default_selections(self):
        """应用默认选择到UI复选框"""
        # 应用默认勾选状态，但不触发事件
        for feat_type, checkbox_widget in self.feature_checkboxes.items():
            states = {}
            for subtype in self.FEATURE_SUBTYPES[feat_type]:
                states[subtype] = self.enabled_features[feat_type].get(subtype, False)
            checkbox_widget.set_states_without_signal(states)

    # 功能实现方法
    def browse_directory(self):
        """浏览选择目录"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择Feature数据目录",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if directory:
            self.path_input.setText(directory)
            self.load_feature_directory()

    def load_feature_directory(self):
        """加载feature目录"""
        dir_path = self.path_input.text().strip()
        if not dir_path:
            QMessageBox.warning(self, "警告", "请先选择数据目录")
            return

        try:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                QMessageBox.warning(self, "警告", f"目录不存在: {dir_path}")
                return

            # 查找所有frame目录
            frame_dirs = sorted([d for d in dir_path.iterdir()
                                 if d.is_dir() and d.name.startswith('frame_')])

            if not frame_dirs:
                QMessageBox.warning(self, "警告", "未找到frame目录")
                return

            self.feature_dir = dir_path
            self.frames = frame_dirs
            self.current_frame_idx = 0

            # 更新UI
            self.frame_slider.setRange(0, len(self.frames) - 1)
            self.frame_spinbox.setRange(0, len(self.frames) - 1)
            self.frame_info_label.setText(f"总帧数: {len(self.frames)}")

            # 加载第一帧
            self.load_frame(0)

            self.statusBar().showMessage(f'成功加载 {len(self.frames)} 帧数据', 3000)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载目录失败: {str(e)}")

    def load_frame(self, frame_idx: int):
        """加载指定帧"""
        if not self.frames or frame_idx < 0 or frame_idx >= len(self.frames):
            return

        frame_dir = self.frames[frame_idx]
        npz_path = frame_dir / 'features.npz'

        if not npz_path.exists():
            QMessageBox.warning(self, "警告", f"未找到features.npz: {frame_dir.name}")
            return

        try:
            # 加载数据
            data = np.load(npz_path)
            self.current_data = dict(data)

            # 预处理数据类型，避免后续重复转换
            for key in list(self.current_data.keys()):
                if isinstance(self.current_data[key], np.ndarray) and self.current_data[key].ndim == 2:
                    # 将所有2D数组转换为合适的类型
                    if key.endswith('_raw') or key.endswith('_full') or key == 'mist':
                        # 确保这些是布尔类型
                        if self.current_data[key].dtype != bool:
                            self.current_data[key] = self.current_data[key].astype(bool)
                    elif key.endswith('_apf'):
                        # 确保APF是浮点类型
                        if self.current_data[key].dtype != float:
                            self.current_data[key] = self.current_data[key].astype(float)

            # 提取元数据
            self.current_metadata = {
                'step': int(data.get('metadata_step', 0)),
                'episode': int(data.get('metadata_episode', 0)),
                'agent_position': data.get('metadata_agent_position', [0, 0]),
                'agent_direction': float(data.get('metadata_agent_direction', 0)),
                'weed_count': int(data.get('metadata_weed_count', 0)),
                'frontier_area': int(data.get('metadata_frontier_area', 0)),
            }

            self.current_frame_idx = frame_idx

            # 打印数据信息用于调试
            print(f"\nLoaded frame {frame_idx}:")
            for key in sorted(self.current_data.keys()):
                if isinstance(self.current_data[key], np.ndarray):
                    data_info = self.current_data[key]
                    if data_info.ndim == 2:
                        print(f"  {key}: shape={data_info.shape}, dtype={data_info.dtype}, "
                              f"range=[{data_info.min():.3f}, {data_info.max():.3f}], "
                              f"non-zero={np.count_nonzero(data_info)}")

            # 更新元数据显示
            self.update_metadata_display()

            # 启用重新计算APF按钮
            if hasattr(self, 'recalc_btn'):
                self.recalc_btn.setEnabled(True)

            # 刷新可视化
            self.refresh_visualization()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载帧数据失败: {str(e)}")

    def update_metadata_display(self):
        """更新元数据显示"""
        if self.current_metadata:
            meta = self.current_metadata
            text = (f"帧 {self.current_frame_idx} | "
                    f"步骤 {meta['step']} | "
                    f"Episode {meta['episode']} | "
                    f"杂草数 {meta['weed_count']} | "
                    f"前沿区域 {meta['frontier_area']}")

            # 添加APF数据源信息
            apf_sources = []
            for feat_type in self.FEATURE_TYPES:
                if self.enabled_features[feat_type].get('apf', False):
                    sources = []
                    if self.enabled_features[feat_type].get('full', False) and f'{feat_type}_full' in self.current_data:
                        sources.append('full')
                    if self.enabled_features[feat_type].get('raw', False) and f'{feat_type}_raw' in self.current_data:
                        sources.append('raw')
                    if sources:
                        apf_sources.append(f"{feat_type}({'+'.join(sources)})")

            if apf_sources:
                text += f" | APF源: {', '.join(apf_sources)}"

            self.metadata_label.setText(text)
            self.metadata_label.setToolTip("当前帧的元数据信息\nAPF源显示了每个APF计算所基于的数据")

    def on_frame_changed(self, value: int):
        """帧滑块变化处理"""
        if value != self.current_frame_idx:
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(value)
            self.frame_spinbox.blockSignals(False)
            self.load_frame(value)

    def on_frame_spinbox_changed(self, value: int):
        """帧数字框变化处理"""
        if value != self.current_frame_idx:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(value)
            self.frame_slider.blockSignals(False)
            self.load_frame(value)

    def jump_to_frame(self, frame_idx: int):
        """跳转到指定帧"""
        if 0 <= frame_idx < len(self.frames):
            self.frame_slider.setValue(frame_idx)

    def on_mode_changed(self, button):
        """可视化模式变化处理"""
        self.aggregate_mode = (self.mode_group.checkedId() == 1)
        self.refresh_visualization()

    def on_feature_toggled(self, feat_type: str, subtype: str, checked: bool):
        """特征复选框变化处理"""
        self.enabled_features[feat_type][subtype] = checked

        # 只在以下情况下询问是否重新计算APF：
        # 1. 改变的是full或raw
        # 2. APF也被勾选
        # 3. 当前有数据加载
        # 4. 不是在初始化阶段
        if (subtype in ['full', 'raw'] and
                self.enabled_features[feat_type].get('apf', False) and
                self.current_data and
                hasattr(self, 'initialization_complete')):

            # reply = QMessageBox.question(
            #     self,
            #     '重新计算APF？',
            #     f'{feat_type}的{subtype}选项已改变，是否需要重新计算APF？\n'
            #     f'（基于当前勾选的full/raw选项）',
            #     QMessageBox.Yes | QMessageBox.No,
            #     QMessageBox.Yes
            # )
            reply = QMessageBox.Yes
            if reply == QMessageBox.Yes:
                self.recalculate_apf()
        else:
            self.refresh_visualization()

    def on_special_feature_toggled(self, feature: str, checked: bool):
        """特殊特征复选框变化处理"""
        self.enabled_features[feature] = checked
        self.refresh_visualization()

    def on_apf_value_changed(self, feat_type: str, value: int):
        """APF参数变化处理"""
        self.apf_params[feat_type] = value

    @staticmethod
    def get_discounted_apf(map_apf: np.ndarray,
                           max_step: int,
                           eps: Optional[float] = None,
                           pad: bool = False) -> np.ndarray:
        """计算折扣APF"""
        if pad:
            map_apf = np.pad(map_apf,
                             pad_width=[[1, 1], [1, 1]],
                             mode='constant',
                             constant_values=(1, 1))
        map_apf, is_empty = cpu_apf_bool(map_apf)
        if not is_empty:
            gamma = (max_step - 1) / max_step
            map_apf = gamma ** map_apf
            if eps is None:
                eps = gamma ** max_step
            map_apf = np.where(map_apf < eps, 0., map_apf)
        if pad:
            map_apf = map_apf[1:-1, 1:-1]
        return map_apf

    def recalculate_apf(self):
        """重新计算APF"""
        if not self.current_data:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        # 检查是否有任何APF被勾选
        has_apf_enabled = any(
            self.enabled_features[feat_type].get('apf', False)
            for feat_type in self.FEATURE_TYPES
        )

        if not has_apf_enabled:
            QMessageBox.information(self, "提示", "请先勾选需要计算的APF特征")
            return

        # 检查是否有对应的数据源被勾选
        missing_sources = []
        for feat_type in self.FEATURE_TYPES:
            if self.enabled_features[feat_type].get('apf', False):
                has_source = (
                        ('full' in self.FEATURE_SUBTYPES[feat_type] and
                         self.enabled_features[feat_type].get('full', False)) or
                        self.enabled_features[feat_type].get('raw', False)
                )
                if not has_source:
                    missing_sources.append(feat_type)

        if missing_sources:
            QMessageBox.warning(
                self,
                "警告",
                f"以下特征的APF已勾选，但没有选择数据源（full或raw）：\n"
                f"{', '.join(missing_sources)}\n\n"
                f"请先勾选对应的full或raw数据源"
            )
            return

        try:
            # 创建进度对话框
            progress = QProgressDialog("正在计算APF...", "取消", 0, len(self.FEATURE_TYPES), self)
            progress.setWindowTitle("计算中")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            # 获取mist用于掩码
            mist = self.current_data.get('mist', np.ones_like(self.current_data.get('frontier_raw', [])))

            # 预先计算frontier的合并地图，供weed使用
            frontier_combined_map = None
            if 'frontier' in [ft for ft in self.FEATURE_TYPES if any(self.enabled_features[ft].values())]:
                # 根据用户当前的frontier选择来构建合并的frontier地图
                frontier_base_map = None

                # 检查frontier的full是否被勾选
                if ('full' in self.FEATURE_SUBTYPES['frontier'] and
                        self.enabled_features['frontier'].get('full', False)):
                    full_key = 'frontier_full'
                    if full_key in self.current_data:
                        frontier_base_map = self.current_data[full_key].copy()

                # 检查frontier的raw是否被勾选
                if self.enabled_features['frontier'].get('raw', False):
                    raw_key = 'frontier_raw'
                    if raw_key in self.current_data:
                        if frontier_base_map is None:
                            frontier_base_map = self.current_data[raw_key].copy()
                        else:
                            # 如果同时勾选了full和raw，进行逻辑或操作
                            frontier_base_map = np.logical_or(frontier_base_map, self.current_data[raw_key])

                # 如果有frontier基础地图，保存合并结果供weed使用
                if frontier_base_map is not None:
                    frontier_combined_map = frontier_base_map.copy()

            # 重新计算每个APF
            for i, feat_type in enumerate(self.FEATURE_TYPES):
                # 更新进度
                progress.setValue(i)
                progress.setLabelText(f"正在计算 {feat_type} APF...")
                QApplication.processEvents()

                if progress.wasCanceled():
                    break

                # 首先获取基础地图数据（根据用户勾选的选项）
                base_map = None

                # 检查full是否被勾选
                if 'full' in self.FEATURE_SUBTYPES[feat_type] and self.enabled_features[feat_type].get('full', False):
                    full_key = f'{feat_type}_full'
                    if full_key in self.current_data:
                        base_map = self.current_data[full_key].copy()

                # 检查raw是否被勾选
                if self.enabled_features[feat_type].get('raw', False):
                    raw_key = f'{feat_type}_raw'
                    if raw_key in self.current_data:
                        if base_map is None:
                            base_map = self.current_data[raw_key].copy()
                        else:
                            # 如果同时勾选了full和raw，进行逻辑或操作
                            base_map = np.logical_or(base_map, self.current_data[raw_key])

                # 如果没有基础地图数据，跳过
                if base_map is None:
                    continue

                # 根据环境逻辑处理不同的特征类型
                if feat_type == 'frontier':
                    # frontier需要应用total_variation_mat并与mist做与操作
                    # processed_map = np.logical_and(total_variation_mat(base_map), mist)
                    processed_map = total_variation_mat(base_map)
                elif feat_type == 'obstacle':
                    # obstacle需要应用total_variation_mat并与mist做与操作
                    # processed_map = np.logical_and(total_variation_mat(base_map), mist)
                    processed_map = total_variation_mat(base_map)
                elif feat_type == 'weed':
                    # weed需要排除frontier区域
                    # 修复：使用用户当前选择的frontier合并地图
                    if frontier_combined_map is not None:
                        processed_map = np.logical_and(base_map, np.logical_not(frontier_combined_map))
                    else:
                        # 如果没有frontier数据被选择，则使用原始weed数据
                        processed_map = base_map
                elif feat_type == 'trajectory':
                    # trajectory直接使用
                    processed_map = base_map
                else:
                    processed_map = base_map

                # 计算APF
                max_step = self.apf_params[feat_type]
                eps = self.DEFAULT_APF_EPS[feat_type]
                pad = self.apf_pad_obstacle if feat_type == 'obstacle' else False

                # 确保输入是浮点类型
                processed_map = processed_map.astype(float)

                apf_data = self.get_discounted_apf(processed_map, max_step, eps, pad)

                # 特殊处理obstacle：确保原始obstacle位置的值
                if feat_type == 'obstacle':
                    # 将原始obstacle位置（与mist的交集）设为最大值
                    obstacle_mask = np.logical_and(base_map, mist)
                    apf_data = np.maximum(apf_data, obstacle_mask.astype(float))

                # 更新数据
                apf_key = f'{feat_type}_apf'
                self.current_data[apf_key] = apf_data

                print(f"Recalculated APF for {feat_type}: shape={apf_data.shape}, "
                      f"non-zero={np.count_nonzero(apf_data)}, max={apf_data.max():.3f}")

                # 调试信息：显示基础地图的统计
                if base_map is not None:
                    print(f"  Base map for {feat_type}: non-zero={np.count_nonzero(base_map)}")
                if feat_type == 'weed' and frontier_combined_map is not None:
                    print(f"  Frontier combined map: non-zero={np.count_nonzero(frontier_combined_map)}")
                    print(f"  Processed weed map: non-zero={np.count_nonzero(processed_map)}")

            # 完成进度
            progress.setValue(len(self.FEATURE_TYPES))
            progress.close()

            # 更新元数据显示
            self.update_metadata_display()

            # 刷新可视化
            self.refresh_visualization()
            self.statusBar().showMessage("APF重新计算完成", 3000)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"计算APF失败: {str(e)}")

    # def recalculate_apf(self):
    #     """重新计算APF"""
    #     if not self.current_data:
    #         QMessageBox.warning(self, "警告", "请先加载数据")
    #         return
    #
    #     # 检查是否有任何APF被勾选
    #     has_apf_enabled = any(
    #         self.enabled_features[feat_type].get('apf', False)
    #         for feat_type in self.FEATURE_TYPES
    #     )
    #
    #     if not has_apf_enabled:
    #         QMessageBox.information(self, "提示", "请先勾选需要计算的APF特征")
    #         return
    #
    #     # 检查是否有对应的数据源被勾选
    #     missing_sources = []
    #     for feat_type in self.FEATURE_TYPES:
    #         if self.enabled_features[feat_type].get('apf', False):
    #             has_source = (
    #                     ('full' in self.FEATURE_SUBTYPES[feat_type] and
    #                      self.enabled_features[feat_type].get('full', False)) or
    #                     self.enabled_features[feat_type].get('raw', False)
    #             )
    #             if not has_source:
    #                 missing_sources.append(feat_type)
    #
    #     if missing_sources:
    #         QMessageBox.warning(
    #             self,
    #             "警告",
    #             f"以下特征的APF已勾选，但没有选择数据源（full或raw）：\n"
    #             f"{', '.join(missing_sources)}\n\n"
    #             f"请先勾选对应的full或raw数据源"
    #         )
    #         return
    #
    #     try:
    #         # 创建进度对话框
    #         progress = QProgressDialog("正在计算APF...", "取消", 0, len(self.FEATURE_TYPES), self)
    #         progress.setWindowTitle("计算中")
    #         progress.setWindowModality(Qt.WindowModal)
    #         progress.setMinimumDuration(0)
    #         progress.setValue(0)
    #
    #         # 获取mist用于掩码
    #         mist = self.current_data.get('mist', np.ones_like(self.current_data.get('frontier_raw', [])))
    #
    #         # 重新计算每个APF
    #         for i, feat_type in enumerate(self.FEATURE_TYPES):
    #             # 更新进度
    #             progress.setValue(i)
    #             progress.setLabelText(f"正在计算 {feat_type} APF...")
    #             QApplication.processEvents()
    #
    #             if progress.wasCanceled():
    #                 break
    #
    #             # 首先获取基础地图数据（根据用户勾选的选项）
    #             base_map = None
    #
    #             # 检查full是否被勾选
    #             if 'full' in self.FEATURE_SUBTYPES[feat_type] and self.enabled_features[feat_type].get('full', False):
    #                 full_key = f'{feat_type}_full'
    #                 if full_key in self.current_data:
    #                     base_map = self.current_data[full_key].copy()
    #
    #             # 检查raw是否被勾选
    #             if self.enabled_features[feat_type].get('raw', False):
    #                 raw_key = f'{feat_type}_raw'
    #                 if raw_key in self.current_data:
    #                     if base_map is None:
    #                         base_map = self.current_data[raw_key].copy()
    #                     else:
    #                         # 如果同时勾选了full和raw，进行逻辑或操作
    #                         base_map = np.logical_or(base_map, self.current_data[raw_key])
    #
    #             # 如果没有基础地图数据，跳过
    #             if base_map is None:
    #                 continue
    #
    #             # 根据环境逻辑处理不同的特征类型
    #             if feat_type == 'frontier':
    #                 # frontier需要应用total_variation_mat并与mist做与操作
    #                 processed_map = np.logical_and(total_variation_mat(base_map), mist)
    #             elif feat_type == 'obstacle':
    #                 # obstacle需要应用total_variation_mat并与mist做与操作
    #                 processed_map = np.logical_and(total_variation_mat(base_map), mist)
    #             elif feat_type == 'weed':
    #                 # weed需要排除frontier区域
    #                 if 'frontier_full' in self.current_data or 'frontier_raw' in self.current_data:
    #                     # 获取当前的frontier地图（优先使用full，如果没有则使用raw）
    #                     frontier_map = self.current_data.get('frontier_full',
    #                                                          self.current_data.get('frontier_raw',
    #                                                                                np.zeros_like(base_map)))
    #                     processed_map = np.logical_and(base_map, np.logical_not(frontier_map))
    #                 else:
    #                     processed_map = base_map
    #             elif feat_type == 'trajectory':
    #                 # trajectory直接使用
    #                 processed_map = base_map
    #             else:
    #                 processed_map = base_map
    #
    #             # 计算APF
    #             max_step = self.apf_params[feat_type]
    #             eps = self.DEFAULT_APF_EPS[feat_type]
    #             pad = self.apf_pad_obstacle if feat_type == 'obstacle' else False
    #
    #             # 确保输入是浮点类型
    #             processed_map = processed_map.astype(float)
    #
    #             apf_data = self.get_discounted_apf(processed_map, max_step, eps, pad)
    #
    #             # 特殊处理obstacle：确保原始obstacle位置的值
    #             if feat_type == 'obstacle':
    #                 # 将原始obstacle位置（与mist的交集）设为最大值
    #                 obstacle_mask = np.logical_and(base_map, mist)
    #                 apf_data = np.maximum(apf_data, obstacle_mask.astype(float))
    #
    #             # 更新数据
    #             apf_key = f'{feat_type}_apf'
    #             self.current_data[apf_key] = apf_data
    #
    #             print(f"Recalculated APF for {feat_type}: shape={apf_data.shape}, "
    #                   f"non-zero={np.count_nonzero(apf_data)}, max={apf_data.max():.3f}")
    #
    #         # 完成进度
    #         progress.setValue(len(self.FEATURE_TYPES))
    #         progress.close()
    #
    #         # 更新元数据显示
    #         self.update_metadata_display()
    #
    #         # 刷新可视化
    #         self.refresh_visualization()
    #         self.statusBar().showMessage("APF重新计算完成", 3000)
    #
    #     except Exception as e:
    #         import traceback
    #         traceback.print_exc()
    #         QMessageBox.critical(self, "错误", f"计算APF失败: {str(e)}")

    def refresh_visualization(self):
        """刷新可视化"""
        if not self.current_data:
            return

        # 清除画布
        self.canvas.fig.clear()

        # 根据模式创建可视化
        if self.aggregate_mode:
            self.create_aggregated_visualization()
        else:
            self.create_separated_visualization()

        # 刷新画布
        self.canvas.draw()

    def create_separated_visualization(self):
        """创建分离模式可视化"""
        # 计算需要的行数
        rows = []

        # 检查每个特征类型
        for feat_type in self.FEATURE_TYPES:
            if any(self.enabled_features[feat_type].values()):
                rows.append(feat_type)

        # 检查特殊特征
        if self.enabled_features['mist'] or self.enabled_features['rendered_map']:
            rows.append('special')

        if not rows:
            return

        # 创建子图
        n_rows = len(rows)
        n_cols = 3  # full, raw, apf

        for i, row_type in enumerate(rows):
            if row_type == 'special':
                # 特殊特征行
                col = 0
                if self.enabled_features['mist'] and 'mist' in self.current_data:
                    ax = self.canvas.fig.add_subplot(n_rows, n_cols, i * n_cols + col + 1)
                    self.visualize_single_feature(ax, 'Mist', self.current_data['mist'])
                    col += 1

                if self.enabled_features['rendered_map'] and 'rendered_map' in self.current_data:
                    ax = self.canvas.fig.add_subplot(n_rows, n_cols, i * n_cols + col + 1)
                    self.visualize_single_feature(ax, 'Rendered Map', self.current_data['rendered_map'])
            else:
                # 常规特征行
                for j, subtype in enumerate(['full', 'raw', 'apf']):
                    if subtype not in self.FEATURE_SUBTYPES[row_type]:
                        continue

                    if self.enabled_features[row_type].get(subtype, False):
                        col_idx = 0 if subtype == 'full' else (1 if subtype == 'raw' else 2)
                        ax = self.canvas.fig.add_subplot(n_rows, n_cols, i * n_cols + col_idx + 1)

                        data_key = f'{row_type}_{subtype}'
                        if data_key in self.current_data:
                            title = f'{row_type.capitalize()} ({subtype.upper()})'
                            self.visualize_single_feature(ax, title, self.current_data[data_key])

    def create_aggregated_visualization(self):
        """创建聚合模式可视化"""
        # 检查是否有特征被启用
        has_features = any(
            any(self.enabled_features[feat_type].values())
            for feat_type in self.FEATURE_TYPES
        )

        if not has_features and not self.enabled_features['mist'] and not self.enabled_features['rendered_map']:
            return

        # 计算布局
        n_rows = 1
        if self.enabled_features['mist'] or self.enabled_features['rendered_map']:
            n_rows = 2

        # 创建子图
        if has_features:
            # Sparse可视化
            ax_sparse = self.canvas.fig.add_subplot(n_rows, 2, 1)
            self.visualize_sparse_aggregate(ax_sparse)

            # MSDF可视化
            ax_msdf = self.canvas.fig.add_subplot(n_rows, 2, 2)
            self.visualize_msdf_aggregate(ax_msdf)

        # 特殊特征
        if n_rows == 2:
            col = 0
            if self.enabled_features['mist'] and 'mist' in self.current_data:
                ax = self.canvas.fig.add_subplot(n_rows, 2, 3 + col)
                self.visualize_single_feature(ax, 'Mist', self.current_data['mist'])
                col += 1

            if self.enabled_features['rendered_map'] and 'rendered_map' in self.current_data:
                ax = self.canvas.fig.add_subplot(n_rows, 2, 3 + col)
                self.visualize_single_feature(ax, 'Rendered Map', self.current_data['rendered_map'])

    def visualize_single_feature(self, ax, title: str, data: Optional[np.ndarray]):
        """可视化单个特征"""
        ax.clear()

        if data is None:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            return

        # 处理不同的数据类型
        if title == 'Rendered Map' and data.ndim == 3:
            # RGB图像
            ax.imshow(data.astype(np.uint8))
        elif 'APF' in title:
            # APF数据，使用热图
            im = ax.imshow(data, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
            cbar = self.canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            # 添加数据统计信息
            non_zero = np.count_nonzero(data)
            if non_zero > 0:
                stats_text = f"非零: {non_zero}\n最大: {data.max():.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        else:
            # 二值数据
            ax.imshow(data, cmap='gray', vmin=0, vmax=1, interpolation='nearest')

            # 添加数据统计信息
            non_zero = np.count_nonzero(data)
            if non_zero > 0:
                stats_text = f"非零: {non_zero}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    def visualize_sparse_aggregate(self, ax):
        """可视化稀疏聚合（非APF特征）"""
        ax.clear()

        # 获取数据维度
        shape = None
        for key in self.current_data:
            if isinstance(self.current_data[key], np.ndarray) and self.current_data[key].ndim == 2:
                shape = self.current_data[key].shape
                break

        if shape is None:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('Sparse Features', fontsize=12, fontweight='bold')
            ax.axis('off')
            return

        # 创建RGB图像
        img = np.ones((*shape, 3))  # 白色背景

        # 添加每个启用的特征
        for feat_type in self.FEATURE_TYPES:
            color = self.COLORS[feat_type]

            # 添加full和raw特征
            for subtype in ['full', 'raw']:
                if self.enabled_features[feat_type].get(subtype, False):
                    data_key = f'{feat_type}_{subtype}'
                    if data_key in self.current_data:
                        mask = self.current_data[data_key] > 0
                        for c in range(3):
                            img[mask, c] = color[c]

        ax.imshow(img)
        ax.set_title('Sparse Features', fontsize=12, fontweight='bold')
        ax.axis('off')

        # # 添加图例
        # handles = []
        # labels = []
        # for feat_type in self.FEATURE_TYPES:
        #     if any(self.enabled_features[feat_type].get(st, False)
        #            for st in ['full', 'raw']):
        #         handles.append(mpatches.Patch(color=self.COLORS[feat_type]))
        #         labels.append(feat_type.capitalize())
        #
        # if handles:
        #     ax.legend(handles, labels, loc='upper right', fontsize=10,
        #               framealpha=0.9, edgecolor='gray')

    def visualize_msdf_aggregate(self, ax):
        """可视化MSDF聚合（APF特征）"""
        ax.clear()

        # 获取数据维度
        shape = None
        for key in self.current_data:
            if isinstance(self.current_data[key], np.ndarray) and self.current_data[key].ndim == 2:
                shape = self.current_data[key].shape
                break

        if shape is None:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('MSDF Features (APF)', fontsize=12, fontweight='bold')
            ax.axis('off')
            return

        # 创建RGB图像
        img = np.ones((*shape, 3))  # 白色背景

        # 收集APF数据
        apf_data = {}
        for feat_type in self.FEATURE_TYPES:
            if self.enabled_features[feat_type].get('apf', False):
                data_key = f'{feat_type}_apf'
                if data_key in self.current_data:
                    apf_data[feat_type] = self.current_data[data_key]

        # 应用最大值策略处理重叠区域
        for i in range(shape[0]):
            for j in range(shape[1]):
                max_val = 0
                max_type = None

                # 找到该像素的最大APF值
                for feat_type, data in apf_data.items():
                    if data[i, j] > max_val:
                        max_val = data[i, j]
                        max_type = feat_type

                # 应用带渐变的颜色
                if max_type is not None and max_val > 0:
                    color = self.COLORS[max_type]
                    # 在白色和特征颜色之间插值
                    for c in range(3):
                        img[i, j, c] = 1.0 * (1 - max_val) + color[c] * max_val

        ax.imshow(img)
        ax.set_title('MSDF Features (APF)', fontsize=12, fontweight='bold')
        ax.axis('off')

        # # 添加图例
        # handles = []
        # labels = []
        # for feat_type in apf_data:
        #     handles.append(mpatches.Patch(color=self.COLORS[feat_type]))
        #     labels.append(f'{feat_type.capitalize()} APF')
        #
        # if handles:
        #     ax.legend(handles, labels, loc='upper right', fontsize=10,
        #               framealpha=0.9, edgecolor='gray')

    def zoom_in(self):
        """放大视图"""
        # 这里可以实现matplotlib的缩放功能
        pass

    def zoom_out(self):
        """缩小视图"""
        # 这里可以实现matplotlib的缩放功能
        pass

    def zoom_reset(self):
        """重置缩放"""
        # 这里可以实现matplotlib的缩放重置
        pass

    def export_figure(self):
        """导出图片"""
        if not self.current_data or not self.feature_dir:
            QMessageBox.warning(self, "警告", "没有数据可导出")
            return

        # 获取保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = 'aggregated' if self.aggregate_mode else 'separated'
        default_name = f'feature_viz_{mode_suffix}_frame{self.current_frame_idx:06d}_{timestamp}.png'

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            str(self.feature_dir / default_name),
            "PNG图片 (*.png);;PDF文档 (*.pdf);;SVG矢量图 (*.svg)"
        )

        if file_path:
            try:
                self.canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight',
                                        facecolor='white', edgecolor='none')
                self.statusBar().showMessage(f'图片已保存: {Path(file_path).name}', 3000)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def keyPressEvent(self, event):
        """处理键盘事件"""
        if event.key() == Qt.Key_Left:
            # 左箭头：上一帧
            self.jump_to_frame(self.current_frame_idx - 1)
        elif event.key() == Qt.Key_Right:
            # 右箭头：下一帧
            self.jump_to_frame(self.current_frame_idx + 1)
        elif event.key() == Qt.Key_Home:
            # Home：第一帧
            self.jump_to_frame(0)
        elif event.key() == Qt.Key_End:
            # End：最后一帧
            self.jump_to_frame(len(self.frames) - 1)
        elif event.key() == Qt.Key_R:
            # R：刷新
            self.refresh_visualization()
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            # Ctrl+S：保存
            self.export_figure()
        else:
            super().keyPressEvent(event)


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 设置应用程序图标（如果有的话）
    # app.setWindowIcon(QIcon('icon.png'))

    # 创建并显示主窗口
    window = FeatureVisualizerGUI()
    window.show()

    # 如果有命令行参数，自动加载
    if len(sys.argv) > 1:
        window.path_input.setText(sys.argv[1])
        window.load_feature_directory()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()