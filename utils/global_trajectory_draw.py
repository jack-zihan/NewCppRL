#!/usr/bin/env python3
"""
全局地图标注工具 - 优化增强版
支持直观的视图操作和完整的标注功能
2025-07-12: v1在初版基础上优化了UI和交互体验，实现了更直观的地图操作和标注功能，并优化了颜色配色
2025-07-12: v2在v1基础上更新实例编辑，使得实例也可以平移旋转等等编辑操作
2025-07-12: 合并三个版本共称原版
"""

import sys
import os
import cv2
import numpy as np
import json
from datetime import datetime
from enum import Enum
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import math


class EditMode(Enum):
    """编辑模式枚举"""
    VIEW = "查看模式"
    TRANSLATE = "平移模式"
    SCALE = "缩放模式"
    ROTATE = "旋转模式"
    INSTANCE = "实例编辑"
    COORD_QUERY = "坐标查询"
    ADD_CIRCLE = "添加圆形"
    CALIBRATE = "两点校准"


class InstanceEditMode(Enum):
    """实例编辑子模式"""
    SELECT = "选择"
    TRANSLATE = "平移"
    SCALE = "缩放"
    ROTATE = "旋转"
    EDIT_SHAPE = "编辑形状"


class MapGraphicsView(QGraphicsView):
    """自定义图形视图，支持更直观的操作"""

    mousePressed = pyqtSignal(QPointF)
    mouseMoved = pyqtSignal(QPointF)
    mouseReleased = pyqtSignal(QPointF)
    viewChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setDragMode(QGraphicsView.NoDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # 视图状态
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.rotation = 0.0

        # 拖动状态
        self.is_panning = False
        self.pan_start_pos = None
        self.last_mouse_pos = None

        # 操作模式
        self.current_mode = EditMode.VIEW

        # 设置视图样式
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))

    def set_mode(self, mode):
        """设置操作模式"""
        self.current_mode = mode
        self.is_panning = False

    def fit_map_in_view(self):
        """让地图适应视图大小"""
        if self.scene():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
            # 记录当前缩放级别
            transform = self.transform()
            self.zoom_level = transform.m11()

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        scene_pos = self.mapToScene(event.pos())

        if event.button() == Qt.LeftButton:
            if self.current_mode == EditMode.VIEW:
                # 查看模式下启用拖动
                self.is_panning = True
                self.pan_start_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            else:
                # 其他模式传递事件
                self.mousePressed.emit(scene_pos)

        elif event.button() == Qt.MiddleButton:
            # 中键始终用于拖动
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

        self.last_mouse_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.is_panning and self.pan_start_pos:
            # 计算拖动偏移
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()

            # 更新视图位置
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())

            self.viewChanged.emit()
        else:
            # 传递场景坐标
            scene_pos = self.mapToScene(event.pos())
            self.mouseMoved.emit(scene_pos)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            if self.is_panning:
                self.is_panning = False
                self.setCursor(Qt.ArrowCursor)
            else:
                scene_pos = self.mapToScene(event.pos())
                self.mouseReleased.emit(scene_pos)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        # 计算缩放因子
        delta = event.angleDelta().y()
        scale_factor = 1.15 if delta > 0 else 1 / 1.15

        # 计算新的缩放级别
        new_zoom = self.zoom_level * scale_factor

        # 限制缩放范围
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.scale(scale_factor, scale_factor)
            self.zoom_level = new_zoom
            self.viewChanged.emit()

    def reset_view(self):
        """重置视图"""
        self.resetTransform()
        self.zoom_level = 1.0
        self.rotation = 0.0
        self.fit_map_in_view()

    def rotate_view(self, angle):
        """旋转视图"""
        self.rotate(angle)
        self.rotation += angle
        self.viewChanged.emit()

    def get_visible_rect(self):
        """获取可见区域的场景矩形"""
        return self.mapToScene(self.viewport().rect()).boundingRect()


class AnnotationGraphicsItem(QGraphicsItem):
    """自定义标注图形项基类"""

    def __init__(self, annotation_type, index):
        super().__init__()
        self.annotation_type = annotation_type
        self.index = index
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.is_hovered = False
        self.instance_transform = QTransform()  # 实例独立的变换

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self.update()

    def get_center(self):
        """获取标注的中心点"""
        bounds = self.boundingRect()
        return bounds.center()


class CircleAnnotation(AnnotationGraphicsItem):
    """圆形标注项"""

    def __init__(self, center, radius, config, index):
        super().__init__('circle', index)
        self.center = center
        self.radius = radius
        self.config = config
        self.is_covered = False

    def boundingRect(self):
        margin = 5
        return QRectF(
            self.center.x() - self.radius - margin,
            self.center.y() - self.radius - margin,
            2 * (self.radius + margin),
            2 * (self.radius + margin)
        )

    def paint(self, painter, option, widget):
        # 应用实例变换
        painter.save()
        painter.setTransform(self.instance_transform, True)

        # 设置画笔和画刷
        fill_color = QColor(*[self.config['circle_fill_color'][i] for i in [2, 1, 0]])
        fill_color.setAlphaF(self.config['circle_alpha'])

        border_color = QColor(*[self.config['circle_border_color'][i] for i in [2, 1, 0]])

        pen = QPen(border_color)
        pen.setWidth(self.config['circle_border_thickness'])

        if self.is_hovered or self.isSelected():
            pen.setWidth(pen.width() + 2)
            fill_color.setAlphaF(min(1.0, fill_color.alphaF() + 0.2))

        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))

        # 绘制圆形
        painter.drawEllipse(self.center, int(self.radius), int(self.radius))

        # 如果被覆盖，绘制红叉
        if self.is_covered:
            cross_pen = QPen(QColor(255, 0, 0))
            cross_pen.setWidth(max(2, int(self.radius * 0.1)))
            painter.setPen(cross_pen)

            cross_size = self.radius * 0.7
            painter.drawLine(
                QPointF(self.center.x() - cross_size, self.center.y() - cross_size),
                QPointF(self.center.x() + cross_size, self.center.y() + cross_size)
            )
            painter.drawLine(
                QPointF(self.center.x() - cross_size, self.center.y() + cross_size),
                QPointF(self.center.x() + cross_size, self.center.y() - cross_size)
            )

        painter.restore()

        # 如果选中，绘制边界框
        if self.isSelected():
            painter.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.boundingRect())

    def get_center(self):
        """获取圆形的中心点"""
        return self.center


class PolygonAnnotation(AnnotationGraphicsItem):
    """多边形标注项"""

    def __init__(self, points, config, index):
        super().__init__('polygon', index)
        self.points = points
        self.config = config
        self.polygon = QPolygonF(points)

    def boundingRect(self):
        return self.polygon.boundingRect().adjusted(-5, -5, 5, 5)

    def paint(self, painter, option, widget):
        # 应用实例变换
        painter.save()
        painter.setTransform(self.instance_transform, True)

        # 设置画笔和画刷
        fill_color = QColor(*[self.config['polygon_fill_color'][i] for i in [2, 1, 0]])
        fill_color.setAlphaF(self.config['polygon_alpha'])

        border_color = QColor(*[self.config['polygon_border_color'][i] for i in [2, 1, 0]])

        pen = QPen(border_color)
        pen.setWidth(self.config['polygon_border_thickness'])

        if self.is_hovered or self.isSelected():
            pen.setWidth(pen.width() + 2)
            fill_color.setAlphaF(min(1.0, fill_color.alphaF() + 0.2))

        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))

        # 绘制多边形
        painter.drawPolygon(self.polygon)

        painter.restore()

        # 如果选中，绘制顶点
        if self.isSelected():
            painter.setPen(QPen(Qt.yellow, 2))
            painter.setBrush(QBrush(Qt.yellow))
            for point in self.points:
                painter.drawEllipse(point, 5, 5)

    def get_center(self):
        """获取多边形的中心点"""
        return self.polygon.boundingRect().center()


class TrajectoryAnnotation(AnnotationGraphicsItem):
    """轨迹标注项"""

    def __init__(self, positions, config):
        super().__init__('trajectory', -1)  # -1 表示不是索引项
        self.positions = positions
        self.config = config
        self.path = QPainterPath()
        self.path.moveTo(positions[0])
        for pos in positions[1:]:
            self.path.lineTo(pos)

    def boundingRect(self):
        return self.path.boundingRect().adjusted(-5, -5, 5, 5)

    def paint(self, painter, option, widget):
        # 应用实例变换
        painter.save()
        painter.setTransform(self.instance_transform, True)

        # 设置画笔
        color = self.config['trajectory_color']
        pen = QPen(QColor(color[2], color[1], color[0]))
        pen.setWidth(self.config['trajectory_thickness'])

        if self.is_hovered or self.isSelected():
            pen.setWidth(pen.width() + 2)
            pen.setColor(QColor(255, 255, 0))

        painter.setPen(pen)
        painter.drawPath(self.path)

        painter.restore()

    def get_center(self):
        """获取轨迹的中心点"""
        return self.path.boundingRect().center()


class GlobalMapAnnotatorGUI(QMainWindow):
    """全局地图标注工具主窗口 - 优化版"""

    def __init__(self):
        super().__init__()

        # 数据路径
        self.tracking_data_path = None
        self.map_image_path = None

        # 数据存储
        self.tracking_data = None
        self.trajectory_positions = []
        self.polygons = []
        self.circles = []
        self.coverage_data = None

        # 图形项
        self.map_item = None
        self.annotation_items = []
        self.helper_items = []  # 辅助显示项

        # 编辑状态
        self.current_mode = EditMode.VIEW
        self.instance_mode = InstanceEditMode.SELECT
        self.selected_item = None
        self.operation_start = None
        self.initial_rotation = 0

        # 实例编辑相关
        self.instance_operation_start = None
        self.instance_initial_state = None

        # 校准相关
        self.calibration_points = []

        # 场景
        self.scene = QGraphicsScene()

        # 顶点编辑相关
        self.vertex_handles = []  # 顶点手柄列表
        self.dragging_vertex = None  # 正在拖动的顶点
        self.dragging_vertex_index = -1  # 正在拖动的顶点索引

        # 初始化UI（这必须在最后，因为UI创建过程中可能会引用上面的属性）
        self.initUI()
        self.setup_styles()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('全局地图标注工具 - 优化增强版')
        self.setGeometry(100, 100, 1400, 900)

        # 创建中心部件
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #e9ecef;")
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # 左侧：地图显示区域
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #f8f9fa;")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 文件选择区域（紧凑布局）
        file_widget = QWidget()
        file_widget.setMaximumHeight(80)
        file_widget.setStyleSheet(
            "background-color: white; border: 1px solid #cccccc; border-radius: 5px; padding: 5px;")
        file_layout = QGridLayout(file_widget)
        file_layout.setSpacing(5)

        file_layout.addWidget(QLabel("数据:"), 0, 0)
        self.data_path_input = QLineEdit()
        file_layout.addWidget(self.data_path_input, 0, 1)
        self.browse_data_btn = QPushButton("浏览")
        self.browse_data_btn.clicked.connect(lambda: self.browse_file('data'))
        file_layout.addWidget(self.browse_data_btn, 0, 2)

        file_layout.addWidget(QLabel("地图:"), 1, 0)
        self.map_path_input = QLineEdit()
        file_layout.addWidget(self.map_path_input, 1, 1)
        self.browse_map_btn = QPushButton("浏览")
        self.browse_map_btn.clicked.connect(lambda: self.browse_file('map'))
        file_layout.addWidget(self.browse_map_btn, 1, 2)

        self.load_btn = QPushButton("加载数据")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                border-color: #28a745;
                font-size: 13px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #218838;
                border-color: #218838;
            }
        """)
        file_layout.addWidget(self.load_btn, 0, 3, 2, 1)
        self.load_btn.clicked.connect(self.load_data)

        left_layout.addWidget(file_widget)

        # 地图视图
        self.view = MapGraphicsView()
        self.view.setScene(self.scene)
        self.view.setStyleSheet("border: 2px solid #cccccc; border-radius: 5px;")
        self.view.mousePressed.connect(self.handle_mouse_press)
        self.view.mouseMoved.connect(self.handle_mouse_move)
        self.view.mouseReleased.connect(self.handle_mouse_release)
        self.view.viewChanged.connect(self.update_view_info)
        left_layout.addWidget(self.view, 1)

        # 视图信息栏
        info_bar = QWidget()
        info_bar.setMaximumHeight(30)
        info_bar.setStyleSheet("background-color: #e5e5e5; border-top: 1px solid #cccccc;")
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(5, 0, 5, 0)

        self.zoom_label = QLabel("缩放: 100%")
        self.zoom_label.setStyleSheet("color: #333333; font-weight: bold;")
        self.rotation_label = QLabel("旋转: 0°")
        self.rotation_label.setStyleSheet("color: #333333; font-weight: bold;")
        self.coord_label = QLabel("坐标: (0, 0)")
        self.coord_label.setStyleSheet("color: #333333; font-weight: bold;")

        info_layout.addWidget(self.zoom_label)
        info_layout.addWidget(self.rotation_label)
        info_layout.addWidget(self.coord_label)
        info_layout.addStretch()

        # 快捷操作按钮
        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                border-color: #6c757d;
                color: white;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5a6268;
                border-color: #5a6268;
            }
        """)
        self.reset_view_btn.clicked.connect(self.reset_view)
        info_layout.addWidget(self.reset_view_btn)

        left_layout.addWidget(info_bar)

        main_layout.addWidget(left_widget, 3)

        # 右侧：控制面板
        self.control_panel = self.create_control_panel()
        main_layout.addWidget(self.control_panel, 1)

        # 状态栏
        self.statusBar().showMessage('准备就绪')

        # 设置快捷键
        self.setup_shortcuts()

    def create_control_panel(self):
        """创建控制面板"""
        panel = QScrollArea()
        panel.setMaximumWidth(400)
        panel.setWidgetResizable(True)
        panel.setStyleSheet("""
            QScrollArea {
                background-color: #f8f9fa;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
        """)

        content = QWidget()
        content.setStyleSheet("background-color: #f8f9fa;")
        layout = QVBoxLayout(content)

        # 1. 模式选择
        mode_group = QGroupBox("操作模式")
        mode_layout = QVBoxLayout()

        self.mode_buttons = {}
        mode_shortcuts = {
            EditMode.VIEW: "V",
            EditMode.TRANSLATE: "T",
            EditMode.SCALE: "S",
            EditMode.ROTATE: "R",
            EditMode.INSTANCE: "I",
            EditMode.COORD_QUERY: "Q",
            EditMode.ADD_CIRCLE: "A",
            EditMode.CALIBRATE: "C"
        }

        for mode in EditMode:
            shortcut = mode_shortcuts.get(mode, "")
            btn_text = f"{mode.value} ({shortcut})" if shortcut else mode.value
            btn = QRadioButton(btn_text)
            btn.toggled.connect(lambda checked, m=mode: self.set_mode(m) if checked else None)
            self.mode_buttons[mode] = btn
            mode_layout.addWidget(btn)
            if mode == EditMode.VIEW:
                btn.setChecked(True)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 2. 变换操作
        transform_group = QGroupBox("标注变换")
        transform_layout = QVBoxLayout()

        # 平移控制
        translate_widget = QWidget()
        translate_layout = QHBoxLayout(translate_widget)
        translate_layout.addWidget(QLabel("平移:"))

        self.translate_x_spin = QDoubleSpinBox()
        self.translate_x_spin.setRange(-10000, 10000)
        self.translate_x_spin.setPrefix("X: ")
        self.translate_x_spin.valueChanged.connect(self.apply_transform)
        translate_layout.addWidget(self.translate_x_spin)

        self.translate_y_spin = QDoubleSpinBox()
        self.translate_y_spin.setRange(-10000, 10000)
        self.translate_y_spin.setPrefix("Y: ")
        self.translate_y_spin.valueChanged.connect(self.apply_transform)
        translate_layout.addWidget(self.translate_y_spin)

        transform_layout.addWidget(translate_widget)

        # 缩放控制
        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        scale_layout.addWidget(QLabel("缩放:"))

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 10.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setSuffix("x")
        self.scale_spin.valueChanged.connect(self.apply_transform)
        scale_layout.addWidget(self.scale_spin)

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 1000)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(lambda v: self.scale_spin.setValue(v / 100))
        scale_layout.addWidget(self.scale_slider)

        transform_layout.addWidget(scale_widget)

        # 旋转控制
        rotate_widget = QWidget()
        rotate_layout = QHBoxLayout(rotate_widget)
        rotate_layout.addWidget(QLabel("旋转:"))

        self.rotate_spin = QDoubleSpinBox()
        self.rotate_spin.setRange(-180, 180)
        self.rotate_spin.setSuffix("°")
        self.rotate_spin.valueChanged.connect(self.apply_transform)
        rotate_layout.addWidget(self.rotate_spin)

        self.rotate_dial = QDial()
        self.rotate_dial.setRange(-180, 180)
        self.rotate_dial.setWrapping(True)
        self.rotate_dial.valueChanged.connect(self.rotate_spin.setValue)
        rotate_layout.addWidget(self.rotate_dial)

        transform_layout.addWidget(rotate_widget)

        # 快速操作按钮
        quick_btn_layout = QHBoxLayout()
        reset_transform_btn = QPushButton("重置变换")
        reset_transform_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                border-color: #ffc107;
                color: #212529;
                font-size: 11px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #e0a800;
                border-color: #e0a800;
            }
        """)
        reset_transform_btn.clicked.connect(self.reset_transform)
        quick_btn_layout.addWidget(reset_transform_btn)

        center_annotations_btn = QPushButton("居中标注")
        center_annotations_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                border-color: #6f42c1;
                font-size: 11px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #5a32a3;
                border-color: #5a32a3;
            }
        """)
        center_annotations_btn.clicked.connect(self.center_annotations)
        quick_btn_layout.addWidget(center_annotations_btn)

        transform_layout.addLayout(quick_btn_layout)

        # 对齐工具按钮
        align_btn_layout = QHBoxLayout()

        show_guides_btn = QPushButton("显示对齐线")
        show_guides_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                border-color: #17a2b8;
                font-size: 11px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #138496;
                border-color: #138496;
            }
        """)
        show_guides_btn.clicked.connect(self.toggle_alignment_guides)
        align_btn_layout.addWidget(show_guides_btn)

        align_horizontal_btn = QPushButton("水平对齐")
        align_horizontal_btn.setStyleSheet("""
            QPushButton {
                background-color: #20c997;
                border-color: #20c997;
                color: white;
                font-size: 11px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #1ba37e;
                border-color: #1ba37e;
            }
        """)
        align_horizontal_btn.clicked.connect(self.align_to_horizontal)
        align_btn_layout.addWidget(align_horizontal_btn)

        transform_layout.addLayout(align_btn_layout)

        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)

        # 3. 实例编辑
        instance_group = QGroupBox("实例编辑")
        instance_layout = QVBoxLayout()

        self.instance_info_label = QLabel("未选中实例")
        instance_layout.addWidget(self.instance_info_label)

        # 实例编辑模式选择
        self.instance_mode_widget = QWidget()
        self.instance_mode_widget.hide()
        instance_mode_layout = QVBoxLayout(self.instance_mode_widget)

        instance_mode_label = QLabel("实例编辑模式:")
        instance_mode_label.setStyleSheet("font-weight: bold; color: #333;")
        instance_mode_layout.addWidget(instance_mode_label)

        self.instance_mode_buttons = {}
        # 暂时不连接信号，等所有控件创建完成后再连接
        for mode in InstanceEditMode:
            btn = QRadioButton(mode.value)
            self.instance_mode_buttons[mode] = btn
            instance_mode_layout.addWidget(btn)
            if mode == InstanceEditMode.SELECT:
                btn.setChecked(True)

        instance_layout.addWidget(self.instance_mode_widget)

        # 实例变换控制
        self.instance_transform_widget = QWidget()
        self.instance_transform_widget.hide()
        instance_transform_layout = QFormLayout(self.instance_transform_widget)

        # 实例平移
        self.instance_translate_x = QDoubleSpinBox()
        self.instance_translate_x.setRange(-10000, 10000)
        self.instance_translate_x.valueChanged.connect(self.apply_instance_transform)
        instance_transform_layout.addRow("平移 X:", self.instance_translate_x)

        self.instance_translate_y = QDoubleSpinBox()
        self.instance_translate_y.setRange(-10000, 10000)
        self.instance_translate_y.valueChanged.connect(self.apply_instance_transform)
        instance_transform_layout.addRow("平移 Y:", self.instance_translate_y)

        # 实例缩放
        self.instance_scale = QDoubleSpinBox()
        self.instance_scale.setRange(0.1, 10.0)
        self.instance_scale.setValue(1.0)
        self.instance_scale.setSingleStep(0.1)
        self.instance_scale.valueChanged.connect(self.apply_instance_transform)
        instance_transform_layout.addRow("缩放:", self.instance_scale)

        # 实例旋转
        self.instance_rotate = QDoubleSpinBox()
        self.instance_rotate.setRange(-180, 180)
        self.instance_rotate.valueChanged.connect(self.apply_instance_transform)
        instance_transform_layout.addRow("旋转°:", self.instance_rotate)

        # 重置实例变换按钮
        reset_instance_btn = QPushButton("重置实例变换")
        reset_instance_btn.clicked.connect(self.reset_instance_transform)
        instance_transform_layout.addRow(reset_instance_btn)

        instance_layout.addWidget(self.instance_transform_widget)

        # 圆形编辑器
        self.circle_editor = self.create_circle_editor()
        self.circle_editor.hide()
        instance_layout.addWidget(self.circle_editor)

        # 多边形编辑器
        self.polygon_editor = self.create_polygon_editor()
        self.polygon_editor.hide()
        instance_layout.addWidget(self.polygon_editor)

        # 删除按钮
        self.delete_btn = QPushButton("删除选中实例")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                border-color: #dc3545;
            }
            QPushButton:hover {
                background-color: #c82333;
                border-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #cccccc;
                color: #666666;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)
        instance_layout.addWidget(self.delete_btn)

        instance_group.setLayout(instance_layout)
        layout.addWidget(instance_group)

        # 4. 数据操作
        data_group = QGroupBox("数据操作")
        data_layout = QVBoxLayout()

        save_btn = QPushButton("保存标注数据")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                border-color: #28a745;
            }
            QPushButton:hover {
                background-color: #218838;
                border-color: #218838;
            }
        """)
        save_btn.clicked.connect(self.save_annotations)
        data_layout.addWidget(save_btn)

        export_btn = QPushButton("导出标注图片")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                border-color: #17a2b8;
            }
            QPushButton:hover {
                background-color: #138496;
                border-color: #138496;
            }
        """)
        export_btn.clicked.connect(self.export_annotated_image)
        data_layout.addWidget(export_btn)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        layout.addStretch()

        panel.setWidget(content)

        # 在所有控件创建完成后连接实例模式信号
        self.instance_mode_buttons[InstanceEditMode.SELECT].toggled.connect(
            lambda checked: self.set_instance_mode(InstanceEditMode.SELECT) if checked else None)
        self.instance_mode_buttons[InstanceEditMode.TRANSLATE].toggled.connect(
            lambda checked: self.set_instance_mode(InstanceEditMode.TRANSLATE) if checked else None)
        self.instance_mode_buttons[InstanceEditMode.SCALE].toggled.connect(
            lambda checked: self.set_instance_mode(InstanceEditMode.SCALE) if checked else None)
        self.instance_mode_buttons[InstanceEditMode.ROTATE].toggled.connect(
            lambda checked: self.set_instance_mode(InstanceEditMode.ROTATE) if checked else None)
        self.instance_mode_buttons[InstanceEditMode.EDIT_SHAPE].toggled.connect(
            lambda checked: self.set_instance_mode(InstanceEditMode.EDIT_SHAPE) if checked else None)

        return panel

    def create_polygon_editor(self):
        """创建多边形编辑器"""
        widget = QWidget()
        widget.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px;")
        layout = QVBoxLayout(widget)

        # 顶点列表
        vertex_label = QLabel("顶点列表:")
        layout.addWidget(vertex_label)

        # 创建顶点列表控件
        self.vertex_list = QListWidget()
        self.vertex_list.setMaximumHeight(150)
        self.vertex_list.currentRowChanged.connect(self.on_vertex_selected)
        layout.addWidget(self.vertex_list)

        # 顶点坐标编辑
        coord_widget = QWidget()
        coord_layout = QHBoxLayout(coord_widget)
        coord_layout.setContentsMargins(0, 0, 0, 0)

        coord_layout.addWidget(QLabel("X:"))
        self.vertex_x_input = QDoubleSpinBox()
        self.vertex_x_input.setRange(-10000, 10000)
        self.vertex_x_input.valueChanged.connect(self.update_selected_vertex)
        coord_layout.addWidget(self.vertex_x_input)

        coord_layout.addWidget(QLabel("Y:"))
        self.vertex_y_input = QDoubleSpinBox()
        self.vertex_y_input.setRange(-10000, 10000)
        self.vertex_y_input.valueChanged.connect(self.update_selected_vertex)
        coord_layout.addWidget(self.vertex_y_input)

        layout.addWidget(coord_widget)

        # 操作按钮
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        add_vertex_btn = QPushButton("添加顶点")
        add_vertex_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                border-color: #28a745;
                font-size: 11px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #218838;
                border-color: #218838;
            }
        """)
        add_vertex_btn.clicked.connect(self.add_polygon_vertex)
        btn_layout.addWidget(add_vertex_btn)

        delete_vertex_btn = QPushButton("删除顶点")
        delete_vertex_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                border-color: #dc3545;
                font-size: 11px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #c82333;
                border-color: #c82333;
            }
        """)
        delete_vertex_btn.clicked.connect(self.delete_polygon_vertex)
        btn_layout.addWidget(delete_vertex_btn)

        layout.addWidget(btn_widget)

        # 应用按钮
        apply_btn = QPushButton("应用修改")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border-color: #0078d4;
                font-size: 11px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #106ebe;
                border-color: #106ebe;
            }
        """)
        apply_btn.clicked.connect(self.apply_polygon_edit)
        layout.addWidget(apply_btn)

        return widget

    def create_circle_editor(self):
        """创建圆形编辑器"""
        widget = QWidget()
        widget.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px;")
        layout = QFormLayout(widget)

        self.circle_x_input = QDoubleSpinBox()
        self.circle_x_input.setRange(-10000, 10000)
        layout.addRow("中心 X:", self.circle_x_input)

        self.circle_y_input = QDoubleSpinBox()
        self.circle_y_input.setRange(-10000, 10000)
        layout.addRow("中心 Y:", self.circle_y_input)

        self.circle_radius_input = QDoubleSpinBox()
        self.circle_radius_input.setRange(1, 1000)
        self.circle_radius_input.setValue(30)
        layout.addRow("半径:", self.circle_radius_input)

        apply_btn = QPushButton("应用修改")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border-color: #0078d4;
                font-size: 11px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #106ebe;
                border-color: #106ebe;
            }
        """)
        apply_btn.clicked.connect(self.apply_circle_edit)
        layout.addRow(apply_btn)

        return widget

    def setup_shortcuts(self):
        """设置快捷键"""
        shortcuts = {
            'V': lambda: self.mode_buttons[EditMode.VIEW].setChecked(True),
            'T': lambda: self.mode_buttons[EditMode.TRANSLATE].setChecked(True),
            'S': lambda: self.mode_buttons[EditMode.SCALE].setChecked(True),
            'R': lambda: self.mode_buttons[EditMode.ROTATE].setChecked(True),
            'I': lambda: self.mode_buttons[EditMode.INSTANCE].setChecked(True),
            'Q': lambda: self.mode_buttons[EditMode.COORD_QUERY].setChecked(True),
            'A': lambda: self.mode_buttons[EditMode.ADD_CIRCLE].setChecked(True),
            'C': lambda: self.mode_buttons[EditMode.CALIBRATE].setChecked(True),
            'Escape': lambda: self.mode_buttons[EditMode.VIEW].setChecked(True),
            'Delete': self.delete_selected,
            'Ctrl+S': self.save_annotations,
            'Ctrl+E': self.export_annotated_image,
            'Space': self.reset_view,
        }

        for key, func in shortcuts.items():
            QShortcut(QKeySequence(key), self, func)

    def setup_styles(self):
        """设置界面样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: 1px solid #0078d4;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #106ebe;
                border-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
                border-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #cccccc;
                color: #666666;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: white;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 4px;
                border-radius: 3px;
                font-size: 12px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #0078d4;
            }
            QGroupBox {
                color: #333333;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                font-size: 13px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: white;
            }
            QRadioButton {
                color: #333333;
                font-size: 12px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #cccccc;
                border-radius: 7px;
                background-color: white;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #0078d4;
                border-radius: 7px;
                background-color: #0078d4;
            }
            QRadioButton::indicator:checked::hover {
                border-color: #106ebe;
                background-color: #106ebe;
            }
            QScrollArea {
                background-color: #f5f5f5;
                border: none;
            }
            QSlider::groove:horizontal {
                border: 1px solid #cccccc;
                height: 6px;
                background: #e5e5e5;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
                border-color: #106ebe;
            }
            QDial {
                background-color: white;
            }
            QStatusBar {
                background-color: #e5e5e5;
                color: #333333;
            }
            QWidget {
                font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
            }
            /* 滚动条样式 */
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #cccccc;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #999999;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def browse_file(self, file_type):
        """浏览文件"""
        if file_type == 'data':
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择追踪数据文件", "", "JSON文件 (*.json)")
            if file_path:
                self.data_path_input.setText(file_path)
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择地图图片", "", "图片文件 (*.png *.jpg *.jpeg)")
            if file_path:
                self.map_path_input.setText(file_path)

    def load_data(self):
        """加载数据和地图"""
        data_path = self.data_path_input.text()
        map_path = self.map_path_input.text()

        if not os.path.exists(data_path) or not os.path.exists(map_path):
            QMessageBox.warning(self, "警告", "请选择有效的文件路径")
            return

        try:
            # 加载追踪数据
            with open(data_path, 'r', encoding='utf-8') as f:
                self.tracking_data = json.load(f)

            self.tracking_data_path = data_path
            self.map_image_path = map_path

            # 清空场景
            self.scene.clear()
            self.annotation_items.clear()

            # 加载地图
            self.load_map_image()

            # 提取并显示数据
            self.extract_tracking_data()
            self.create_annotations()

            # 适应视图
            self.view.fit_map_in_view()
            self.center_annotations()

            self.statusBar().showMessage('数据加载成功')

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据失败：{str(e)}")

    def load_map_image(self):
        """加载地图图片"""
        pixmap = QPixmap(self.map_image_path)
        if pixmap.isNull():
            raise ValueError("无法加载地图图片")

        # 添加地图到场景
        self.map_item = self.scene.addPixmap(pixmap)
        self.map_item.setZValue(-1)  # 确保地图在底层

        # 设置场景矩形
        self.scene.setSceneRect(QRectF(pixmap.rect()))

    def extract_tracking_data(self):
        """提取追踪数据"""
        trajectory_data = self.tracking_data['trajectory_summary']
        annotations = self.tracking_data['annotations']
        self.render_config = self.tracking_data['render_config']

        # 轨迹点
        self.trajectory_positions = [
            QPointF(pos[0], pos[1])
            for pos in trajectory_data['trajectory_positions']
        ]

        # 多边形
        self.polygons = []
        for poly in annotations['polygons']:
            points = [QPointF(p[0], p[1]) for p in poly['points']]
            self.polygons.append({'points': points, 'data': poly})

        # 圆形
        self.circles = []
        for circle in annotations['circles']:
            center = QPointF(circle['center'][0], circle['center'][1])
            self.circles.append({
                'center': center,
                'radius': circle['radius'],
                'data': circle
            })

        # 加载覆盖数据
        self.load_coverage_data()

    def load_coverage_data(self):
        """加载覆盖数据"""
        video_dir = os.path.dirname(self.tracking_data_path)
        coverage_cache_path = os.path.join(video_dir, 'coverage_cache_scale15.npz')

        if os.path.exists(coverage_cache_path):
            try:
                coverage_data = np.load(coverage_cache_path, allow_pickle=True)
                self.coverage_data = {
                    'circle_coverage_frames': coverage_data['circle_coverage_frames'].tolist(),
                    'num_circles': int(coverage_data['num_circles'])
                }
            except Exception as e:
                print(f"加载覆盖数据失败: {e}")
                self.coverage_data = None
        else:
            self.coverage_data = None

    def create_annotations(self):
        """创建标注图形项"""
        # 绘制轨迹
        if len(self.trajectory_positions) > 1:
            # 创建轨迹图形项
            trajectory_item = TrajectoryAnnotation(
                self.trajectory_positions,
                self.render_config
            )
            self.scene.addItem(trajectory_item)
            self.annotation_items.append(trajectory_item)
            trajectory_item.setZValue(0)

        # 创建多边形
        for i, poly in enumerate(self.polygons):
            item = PolygonAnnotation(poly['points'], self.render_config, i)
            self.scene.addItem(item)
            self.annotation_items.append(item)
            item.setZValue(1)

        # 创建圆形
        for i, circle in enumerate(self.circles):
            item = CircleAnnotation(
                circle['center'],
                circle['radius'],
                self.render_config,
                i
            )

            # 检查是否被覆盖
            if self.coverage_data and i < self.coverage_data['num_circles']:
                if self.coverage_data['circle_coverage_frames'][i] != -1:
                    item.is_covered = True

            self.scene.addItem(item)
            self.annotation_items.append(item)
            item.setZValue(2)

    def set_mode(self, mode):
        """设置编辑模式"""
        self.current_mode = mode
        self.view.set_mode(mode)

        # 清除辅助显示
        self.clear_helpers()

        # 更新光标
        if mode == EditMode.VIEW:
            self.view.setCursor(Qt.OpenHandCursor)
        elif mode == EditMode.TRANSLATE:
            self.view.setCursor(Qt.SizeAllCursor)
        elif mode == EditMode.SCALE:
            self.view.setCursor(Qt.SizeBDiagCursor)
        elif mode == EditMode.ROTATE:
            self.create_rotation_helper()
            self.view.setCursor(Qt.PointingHandCursor)
        elif mode == EditMode.INSTANCE:
            self.view.setCursor(Qt.ArrowCursor)
            # 显示实例模式选择
            if self.selected_item:
                self.instance_mode_widget.show()
        elif mode == EditMode.COORD_QUERY:
            self.view.setCursor(Qt.CrossCursor)
        elif mode == EditMode.ADD_CIRCLE:
            self.view.setCursor(Qt.CrossCursor)
        elif mode == EditMode.CALIBRATE:
            self.view.setCursor(Qt.CrossCursor)
            self.calibration_points = []

        self.statusBar().showMessage(f'当前模式：{mode.value}')

    def set_instance_mode(self, mode):
        """设置实例编辑子模式"""
        self.instance_mode = mode

        # 确保控件已经创建
        if not hasattr(self, 'instance_transform_widget'):
            return

        # 更新UI显示
        if mode in [InstanceEditMode.TRANSLATE, InstanceEditMode.SCALE, InstanceEditMode.ROTATE]:
            self.instance_transform_widget.show()
            self.circle_editor.hide()
            self.polygon_editor.hide()

            # 根据模式设置光标
            if mode == InstanceEditMode.TRANSLATE:
                self.view.setCursor(Qt.SizeAllCursor)
            elif mode == InstanceEditMode.SCALE:
                self.view.setCursor(Qt.SizeBDiagCursor)
            elif mode == InstanceEditMode.ROTATE:
                self.view.setCursor(Qt.PointingHandCursor)

        elif mode == InstanceEditMode.EDIT_SHAPE:
            self.instance_transform_widget.hide()
            if self.selected_item:
                if self.selected_item.annotation_type == 'circle':
                    self.circle_editor.show()
                    self.polygon_editor.hide()
                elif self.selected_item.annotation_type == 'polygon':
                    self.circle_editor.hide()
                    self.polygon_editor.show()
                    self.create_vertex_handles(self.selected_item)
        else:  # SELECT mode
            self.instance_transform_widget.hide()
            self.view.setCursor(Qt.ArrowCursor)

        self.statusBar().showMessage(f'实例编辑模式：{mode.value}')

    def clear_helpers(self):
        """清除辅助显示"""
        for item in self.helper_items:
            self.scene.removeItem(item)
        self.helper_items.clear()

    def toggle_alignment_guides(self):
        """切换对齐辅助线显示"""
        if hasattr(self, 'guides_shown') and self.guides_shown:
            self.clear_helpers()
            self.guides_shown = False
        else:
            self.show_alignment_guides()
            self.guides_shown = True

    def show_alignment_guides(self):
        """显示对齐辅助线和信息"""
        self.clear_helpers()

        # 查找圆形标注
        circles = []
        for item in self.annotation_items:
            if isinstance(item, CircleAnnotation):
                circles.append(item)

        if len(circles) >= 2:
            # 获取前两个圆形的场景坐标
            c1_center = circles[0].mapToScene(circles[0].center)
            c2_center = circles[1].mapToScene(circles[1].center)

            # 绘制连接线
            pen = QPen(QColor(0, 255, 255), 3, Qt.DashLine)
            line = self.scene.addLine(
                c1_center.x(), c1_center.y(),
                c2_center.x(), c2_center.y(), pen
            )
            line.setZValue(100)
            self.helper_items.append(line)

            # 计算距离和角度
            line_vec = QLineF(c1_center, c2_center)
            distance = line_vec.length()
            angle = line_vec.angle()  # 0-360度，0度是正东方向

            # 显示信息背景
            mid_point = (c1_center + c2_center) / 2
            info_text = f"距离: {distance:.1f}\n角度: {angle:.1f}°"

            text_item = self.scene.addText(info_text)
            text_item.setDefaultTextColor(QColor(255, 255, 255))
            font = QFont("Arial", 12, QFont.Bold)
            text_item.setFont(font)

            # 添加背景
            text_rect = text_item.boundingRect()
            bg_rect = self.scene.addRect(text_rect, QPen(Qt.NoPen), QBrush(QColor(0, 0, 0, 180)))
            bg_rect.setPos(mid_point.x() - text_rect.width() / 2, mid_point.y() - text_rect.height() / 2)
            bg_rect.setZValue(99)
            self.helper_items.append(bg_rect)

            text_item.setPos(mid_point.x() - text_rect.width() / 2, mid_point.y() - text_rect.height() / 2)
            text_item.setZValue(100)
            self.helper_items.append(text_item)

            # 绘制角度弧
            arc_radius = 50
            arc_rect = QRectF(
                c1_center.x() - arc_radius,
                c1_center.y() - arc_radius,
                arc_radius * 2,
                arc_radius * 2
            )

            # 用路径绘制角度弧
            path = QPainterPath()
            path.moveTo(c1_center)
            path.arcTo(arc_rect, 0, -angle)
            arc_pen = QPen(QColor(255, 255, 0), 2)
            path_item = self.scene.addPath(path, arc_pen)
            path_item.setZValue(100)
            self.helper_items.append(path_item)

    def align_to_horizontal(self):
        """将标注旋转到水平方向（基于前两个圆形）"""
        circles = []
        for item in self.annotation_items:
            if isinstance(item, CircleAnnotation):
                circles.append(item)

        if len(circles) >= 2:
            # 获取前两个圆形的原始位置（不考虑变换）
            c1_pos = circles[0].center
            c2_pos = circles[1].center

            # 计算当前角度
            dx = c2_pos.x() - c1_pos.x()
            dy = c2_pos.y() - c1_pos.y()
            current_angle = math.degrees(math.atan2(dy, dx))

            # 计算需要旋转的角度使其水平
            rotation_needed = -current_angle

            # 应用旋转
            self.rotate_spin.setValue(rotation_needed)
            self.apply_transform()

            self.statusBar().showMessage(f'已旋转 {rotation_needed:.1f}° 至水平')

            # 更新对齐辅助线
            if hasattr(self, 'guides_shown') and self.guides_shown:
                self.show_alignment_guides()
        else:
            QMessageBox.warning(self, "警告", "需要至少两个圆形标注才能进行水平对齐")

    def create_rotation_helper(self):
        """创建旋转辅助显示"""
        # 获取所有标注的中心
        if not self.annotation_items:
            return

        center = self.get_annotations_center()

        # 绘制中心点
        pen = QPen(QColor(255, 255, 0), 2)
        brush = QBrush(QColor(255, 255, 0))

        center_item = self.scene.addEllipse(
            QRectF(center.x() - 5, center.y() - 5, 10, 10),
            pen, brush
        )
        center_item.setZValue(100)
        self.helper_items.append(center_item)

        # 绘制旋转圆
        radius = 100
        pen.setStyle(Qt.DashLine)
        circle_item = self.scene.addEllipse(
            QRectF(center.x() - radius, center.y() - radius,
                   radius * 2, radius * 2),
            pen
        )
        circle_item.setZValue(100)
        self.helper_items.append(circle_item)

    def get_annotations_center(self):
        """获取所有标注的中心点"""
        if not self.annotation_items:
            return QPointF(0, 0)

        # 计算边界框
        bounds = self.annotation_items[0].boundingRect()
        scene_bounds = self.annotation_items[0].mapRectToScene(bounds)

        for item in self.annotation_items[1:]:
            item_bounds = item.mapRectToScene(item.boundingRect())
            scene_bounds = scene_bounds.united(item_bounds)

        return scene_bounds.center()

    def handle_mouse_press(self, pos):
        """处理鼠标按下事件"""
        if self.current_mode == EditMode.INSTANCE:
            if self.instance_mode == InstanceEditMode.SELECT:
                # 检查是否点击了顶点手柄
                items = self.scene.items(pos)
                for item in items:
                    # 检查是否是顶点手柄
                    if item in self.vertex_handles:
                        self.dragging_vertex = item
                        self.dragging_vertex_index = item.data(0)
                        return

                    # 检查是否是标注项
                    if isinstance(item, AnnotationGraphicsItem):
                        self.select_annotation(item)
                        return

                self.clear_selection()

            elif self.selected_item and self.instance_mode in [InstanceEditMode.TRANSLATE,
                                                               InstanceEditMode.SCALE,
                                                               InstanceEditMode.ROTATE]:
                # 开始实例变换操作
                self.instance_operation_start = pos

                if self.instance_mode == InstanceEditMode.SCALE:
                    self.instance_scale_center = self.selected_item.mapToScene(self.selected_item.get_center())
                elif self.instance_mode == InstanceEditMode.ROTATE:
                    self.instance_rotation_center = self.selected_item.mapToScene(self.selected_item.get_center())

            elif self.instance_mode == InstanceEditMode.EDIT_SHAPE and self.selected_item:
                # 处理形状编辑（顶点拖动等）
                items = self.scene.items(pos)
                for item in items:
                    if item in self.vertex_handles:
                        self.dragging_vertex = item
                        self.dragging_vertex_index = item.data(0)
                        return

        elif self.current_mode == EditMode.TRANSLATE:
            self.operation_start = pos

        elif self.current_mode == EditMode.SCALE:
            self.operation_start = pos
            self.scale_center = self.get_annotations_center()
            self.show_scale_center()

        elif self.current_mode == EditMode.ROTATE:
            self.operation_start = pos
            self.rotation_center = self.get_annotations_center()
            self.initial_rotation = 0

        elif self.current_mode == EditMode.COORD_QUERY:
            self.show_coordinate(pos)

        elif self.current_mode == EditMode.ADD_CIRCLE:
            self.add_circle_at(pos)

        elif self.current_mode == EditMode.CALIBRATE:
            self.add_calibration_point(pos)

    def handle_mouse_move(self, pos):
        """处理鼠标移动事件"""
        # 更新坐标显示
        self.coord_label.setText(f"坐标: ({pos.x():.1f}, {pos.y():.1f})")

        # 处理顶点拖动
        if self.dragging_vertex and self.selected_item and self.selected_item.annotation_type == 'polygon':
            # 更新顶点位置
            polygon = self.polygons[self.selected_item.index]
            polygon['points'][self.dragging_vertex_index] = pos

            # 更新手柄位置
            self.dragging_vertex.setRect(pos.x() - 5, pos.y() - 5, 10, 10)

            # 更新多边形图形
            self.update_polygon_graphics()

            # 更新顶点列表和输入框
            self.load_polygon_vertices(self.selected_item.index)
            self.vertex_list.setCurrentRow(self.dragging_vertex_index)

            return

        # 处理实例变换
        if self.instance_operation_start and self.selected_item and self.current_mode == EditMode.INSTANCE:
            if self.instance_mode == InstanceEditMode.TRANSLATE:
                # 计算平移
                delta = pos - self.instance_operation_start

                # 更新实例变换
                transform = QTransform()
                transform.translate(delta.x(), delta.y())
                self.selected_item.instance_transform = transform
                self.selected_item.update()

                # 更新UI
                self.instance_translate_x.blockSignals(True)
                self.instance_translate_y.blockSignals(True)
                self.instance_translate_x.setValue(delta.x())
                self.instance_translate_y.setValue(delta.y())
                self.instance_translate_x.blockSignals(False)
                self.instance_translate_y.blockSignals(False)

            elif self.instance_mode == InstanceEditMode.SCALE:
                # 计算缩放
                start_dist = QLineF(self.instance_scale_center, self.instance_operation_start).length()
                current_dist = QLineF(self.instance_scale_center, pos).length()

                if start_dist > 0:
                    scale_factor = current_dist / start_dist

                    # 更新实例变换
                    center = self.selected_item.get_center()
                    transform = QTransform()
                    transform.translate(center.x(), center.y())
                    transform.scale(scale_factor, scale_factor)
                    transform.translate(-center.x(), -center.y())
                    self.selected_item.instance_transform = transform
                    self.selected_item.update()

                    # 更新UI
                    self.instance_scale.blockSignals(True)
                    self.instance_scale.setValue(scale_factor)
                    self.instance_scale.blockSignals(False)

            elif self.instance_mode == InstanceEditMode.ROTATE:
                # 计算旋转角度
                start_angle = math.atan2(
                    self.instance_operation_start.y() - self.instance_rotation_center.y(),
                    self.instance_operation_start.x() - self.instance_rotation_center.x()
                )
                current_angle = math.atan2(
                    pos.y() - self.instance_rotation_center.y(),
                    pos.x() - self.instance_rotation_center.x()
                )

                rotation = math.degrees(current_angle - start_angle)

                # 更新实例变换
                center = self.selected_item.get_center()
                transform = QTransform()
                transform.translate(center.x(), center.y())
                transform.rotate(rotation)
                transform.translate(-center.x(), -center.y())
                self.selected_item.instance_transform = transform
                self.selected_item.update()

                # 更新UI
                self.instance_rotate.blockSignals(True)
                self.instance_rotate.setValue(rotation)
                self.instance_rotate.blockSignals(False)

            return

        # 处理全局变换
        if not self.operation_start:
            return

        if self.current_mode == EditMode.TRANSLATE:
            # 计算平移
            delta = pos - self.operation_start

            # 通过变换矩阵实现平移
            transform = QTransform()
            transform.translate(delta.x(), delta.y())

            # 应用到所有标注
            for item in self.annotation_items:
                item.setTransform(transform)

            # 更新平移值显示
            self.translate_x_spin.blockSignals(True)
            self.translate_y_spin.blockSignals(True)
            self.translate_x_spin.setValue(delta.x())
            self.translate_y_spin.setValue(delta.y())
            self.translate_x_spin.blockSignals(False)
            self.translate_y_spin.blockSignals(False)

        elif self.current_mode == EditMode.SCALE:
            # 计算缩放
            start_dist = QLineF(self.scale_center, self.operation_start).length()
            current_dist = QLineF(self.scale_center, pos).length()

            if start_dist > 0:
                scale_factor = current_dist / start_dist

                # 缩放所有标注
                transform = QTransform()
                transform.translate(self.scale_center.x(), self.scale_center.y())
                transform.scale(scale_factor, scale_factor)
                transform.translate(-self.scale_center.x(), -self.scale_center.y())

                for item in self.annotation_items:
                    item.setTransform(transform)

                # 更新缩放值显示
                self.scale_spin.blockSignals(True)
                self.scale_spin.setValue(scale_factor)
                self.scale_spin.blockSignals(False)

        elif self.current_mode == EditMode.ROTATE:
            # 计算旋转角度
            start_angle = math.atan2(
                self.operation_start.y() - self.rotation_center.y(),
                self.operation_start.x() - self.rotation_center.x()
            )
            current_angle = math.atan2(
                pos.y() - self.rotation_center.y(),
                pos.x() - self.rotation_center.x()
            )

            rotation = math.degrees(current_angle - start_angle)

            # 旋转所有标注
            transform = QTransform()
            transform.translate(self.rotation_center.x(), self.rotation_center.y())
            transform.rotate(rotation)
            transform.translate(-self.rotation_center.x(), -self.rotation_center.y())

            for item in self.annotation_items:
                item.setTransform(transform)

            # 更新旋转值显示
            self.rotate_spin.blockSignals(True)
            self.rotate_spin.setValue(rotation)
            self.rotate_spin.blockSignals(False)

    def handle_mouse_release(self, pos):
        """处理鼠标释放事件"""
        if self.dragging_vertex:
            self.dragging_vertex = None
            self.dragging_vertex_index = -1
            self.statusBar().showMessage('顶点位置已更新')
            return

        if self.instance_operation_start:
            # 应用实例变换到数据
            if self.selected_item and self.instance_mode in [InstanceEditMode.TRANSLATE,
                                                             InstanceEditMode.SCALE,
                                                             InstanceEditMode.ROTATE]:
                self.apply_instance_transform_to_data()
            self.instance_operation_start = None

        if self.operation_start:
            # 应用最终的变换到实际数据
            if self.current_mode in [EditMode.TRANSLATE, EditMode.SCALE, EditMode.ROTATE]:
                self.apply_current_transform()

        self.operation_start = None
        self.clear_helpers()

    def apply_instance_transform(self):
        """从UI控件应用实例变换"""
        if not self.selected_item:
            return

        # 获取变换参数
        translate_x = self.instance_translate_x.value()
        translate_y = self.instance_translate_y.value()
        scale = self.instance_scale.value()
        rotation = self.instance_rotate.value()

        # 构建变换矩阵
        center = self.selected_item.get_center()
        transform = QTransform()

        # 应用变换（顺序：平移->缩放->旋转）
        transform.translate(translate_x, translate_y)
        transform.translate(center.x(), center.y())
        transform.scale(scale, scale)
        transform.rotate(rotation)
        transform.translate(-center.x(), -center.y())

        self.selected_item.instance_transform = transform
        self.selected_item.update()

    def apply_instance_transform_to_data(self):
        """将实例变换应用到实际数据"""
        if not self.selected_item:
            return

        transform = self.selected_item.instance_transform

        if self.selected_item.annotation_type == 'circle':
            # 变换圆形中心
            index = self.selected_item.index
            new_center = transform.map(self.selected_item.center)

            # 计算缩放因子
            unit_vector = QPointF(1.0, 0.0)
            transformed_vector = transform.map(unit_vector) - transform.map(QPointF(0, 0))
            scale_factor = math.sqrt(transformed_vector.x() ** 2 + transformed_vector.y() ** 2)

            # 更新数据
            self.circles[index]['center'] = new_center
            self.circles[index]['radius'] = self.selected_item.radius * scale_factor

            # 更新图形项
            self.selected_item.center = new_center
            self.selected_item.radius = self.circles[index]['radius']

        elif self.selected_item.annotation_type == 'polygon':
            # 变换多边形顶点
            index = self.selected_item.index
            new_points = [transform.map(p) for p in self.selected_item.points]

            # 更新数据
            self.polygons[index]['points'] = new_points

            # 更新图形项
            self.selected_item.points = new_points
            self.selected_item.polygon = QPolygonF(new_points)

        elif self.selected_item.annotation_type == 'trajectory':
            # 变换轨迹点
            new_positions = [transform.map(pos) for pos in self.selected_item.positions]

            # 更新数据
            self.trajectory_positions = new_positions

            # 更新图形项
            self.selected_item.positions = new_positions
            self.selected_item.path = QPainterPath()
            self.selected_item.path.moveTo(new_positions[0])
            for pos in new_positions[1:]:
                self.selected_item.path.lineTo(pos)

        # 重置实例变换
        self.selected_item.instance_transform = QTransform()
        self.selected_item.update()

        # 重置UI
        self.reset_instance_transform()

        self.statusBar().showMessage(f'{self.selected_item.annotation_type} 变换已应用')

    def reset_instance_transform(self):
        """重置实例变换"""
        self.instance_translate_x.setValue(0)
        self.instance_translate_y.setValue(0)
        self.instance_scale.setValue(1.0)
        self.instance_rotate.setValue(0)

        if self.selected_item:
            self.selected_item.instance_transform = QTransform()
            self.selected_item.update()

    def show_scale_center(self):
        """显示缩放中心"""
        pen = QPen(QColor(0, 255, 0), 2)
        brush = QBrush(QColor(0, 255, 0))

        center_item = self.scene.addEllipse(
            QRectF(self.scale_center.x() - 5, self.scale_center.y() - 5, 10, 10),
            pen, brush
        )
        center_item.setZValue(100)
        self.helper_items.append(center_item)

    def show_coordinate(self, pos):
        """显示坐标信息"""
        # 创建标记
        pen = QPen(QColor(255, 0, 0), 2)
        marker = self.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen)
        marker.setZValue(100)
        self.helper_items.append(marker)

        # 创建文本
        text = self.scene.addText(f"({pos.x():.1f}, {pos.y():.1f})")
        text.setDefaultTextColor(QColor(255, 255, 255))
        text.setPos(pos + QPointF(10, -20))
        text.setZValue(100)

        # 为文本添加背景
        text_rect = text.boundingRect()
        bg_rect = self.scene.addRect(text_rect, QPen(Qt.NoPen), QBrush(QColor(0, 0, 0, 180)))
        bg_rect.setPos(text.pos())
        bg_rect.setZValue(99)

        self.helper_items.append(bg_rect)
        self.helper_items.append(text)

        self.statusBar().showMessage(f"坐标: ({pos.x():.1f}, {pos.y():.1f})")

    def add_circle_at(self, pos):
        """在指定位置添加圆形"""
        # 创建新圆形数据
        radius = self.circle_radius_input.value()
        new_circle = {
            'center': pos,
            'radius': radius,
            'data': {
                'center': [pos.x(), pos.y()],
                'radius': radius,
                'global_coords': True
            }
        }

        self.circles.append(new_circle)

        # 创建图形项
        item = CircleAnnotation(pos, radius, self.render_config, len(self.circles) - 1)
        self.scene.addItem(item)
        self.annotation_items.append(item)
        item.setZValue(2)

        # 选中新圆形
        self.select_annotation(item)

        self.statusBar().showMessage(f'已添加圆形 #{len(self.circles) - 1}')

    def select_annotation(self, item):
        """选择标注项"""
        # 清除之前的选择
        self.scene.clearSelection()
        self.clear_vertex_handles()  # 清除顶点手柄

        # 选中新项
        item.setSelected(True)
        self.selected_item = item

        # 更新UI
        self.delete_btn.setEnabled(True)

        if item.annotation_type == 'circle':
            self.instance_info_label.setText(f"圆形 #{item.index}")

            # 显示圆形编辑器
            circle = self.circles[item.index]
            self.circle_x_input.setValue(circle['center'].x())
            self.circle_y_input.setValue(circle['center'].y())
            self.circle_radius_input.setValue(circle['radius'])

            if self.instance_mode == InstanceEditMode.EDIT_SHAPE:
                self.circle_editor.show()
                self.polygon_editor.hide()

        elif item.annotation_type == 'polygon':
            self.instance_info_label.setText(f"多边形 #{item.index}")

            # 加载多边形顶点
            self.load_polygon_vertices(item.index)

            if self.instance_mode == InstanceEditMode.EDIT_SHAPE:
                self.circle_editor.hide()
                self.polygon_editor.show()
                # 创建顶点手柄
                self.create_vertex_handles(item)

        elif item.annotation_type == 'trajectory':
            self.instance_info_label.setText("轨迹")
            self.circle_editor.hide()
            self.polygon_editor.hide()

        # 显示实例模式选择
        if self.current_mode == EditMode.INSTANCE:
            self.instance_mode_widget.show()

            # 根据当前实例模式更新UI
            if self.instance_mode in [InstanceEditMode.TRANSLATE, InstanceEditMode.SCALE, InstanceEditMode.ROTATE]:
                self.instance_transform_widget.show()
            elif self.instance_mode == InstanceEditMode.EDIT_SHAPE:
                self.set_instance_mode(InstanceEditMode.EDIT_SHAPE)

    def load_polygon_vertices(self, polygon_index):
        """加载多边形顶点到列表"""
        self.vertex_list.clear()
        polygon = self.polygons[polygon_index]

        for i, point in enumerate(polygon['points']):
            self.vertex_list.addItem(f"顶点 {i + 1}: ({point.x():.1f}, {point.y():.1f})")

        if self.vertex_list.count() > 0:
            self.vertex_list.setCurrentRow(0)

    def on_vertex_selected(self, row):
        """当选择顶点时更新坐标输入框"""
        if row < 0 or not self.selected_item or self.selected_item.annotation_type != 'polygon':
            return

        polygon = self.polygons[self.selected_item.index]
        if row < len(polygon['points']):
            point = polygon['points'][row]
            self.vertex_x_input.blockSignals(True)
            self.vertex_y_input.blockSignals(True)
            self.vertex_x_input.setValue(point.x())
            self.vertex_y_input.setValue(point.y())
            self.vertex_x_input.blockSignals(False)
            self.vertex_y_input.blockSignals(False)

            # 高亮显示选中的顶点
            self.highlight_vertex(row)

    def update_selected_vertex(self):
        """更新选中顶点的坐标"""
        row = self.vertex_list.currentRow()
        if row < 0 or not self.selected_item or self.selected_item.annotation_type != 'polygon':
            return

        polygon = self.polygons[self.selected_item.index]
        if row < len(polygon['points']):
            # 更新坐标
            new_x = self.vertex_x_input.value()
            new_y = self.vertex_y_input.value()
            polygon['points'][row] = QPointF(new_x, new_y)

            # 更新列表显示
            self.vertex_list.item(row).setText(f"顶点 {row + 1}: ({new_x:.1f}, {new_y:.1f})")

            # 更新图形
            self.update_polygon_graphics()

    def add_polygon_vertex(self):
        """添加新顶点"""
        if not self.selected_item or self.selected_item.annotation_type != 'polygon':
            return

        polygon = self.polygons[self.selected_item.index]
        current_row = self.vertex_list.currentRow()

        # 在当前选中顶点后插入新顶点
        if current_row >= 0 and current_row < len(polygon['points']) - 1:
            # 在当前顶点和下一个顶点中间插入
            p1 = polygon['points'][current_row]
            p2 = polygon['points'][current_row + 1]
            new_point = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
            polygon['points'].insert(current_row + 1, new_point)
        else:
            # 在末尾添加（靠近第一个顶点）
            if len(polygon['points']) > 0:
                last = polygon['points'][-1]
                first = polygon['points'][0]
                new_point = QPointF((last.x() + first.x()) / 2, (last.y() + first.y()) / 2)
                polygon['points'].append(new_point)

        # 重新加载顶点列表
        self.load_polygon_vertices(self.selected_item.index)

        # 更新图形
        self.update_polygon_graphics()
        self.create_vertex_handles(self.selected_item)

    def delete_polygon_vertex(self):
        """删除选中的顶点"""
        if not self.selected_item or self.selected_item.annotation_type != 'polygon':
            return

        polygon = self.polygons[self.selected_item.index]
        current_row = self.vertex_list.currentRow()

        # 至少保留3个顶点
        if current_row >= 0 and len(polygon['points']) > 3:
            polygon['points'].pop(current_row)

            # 重新加载顶点列表
            self.load_polygon_vertices(self.selected_item.index)

            # 更新图形
            self.update_polygon_graphics()
            self.create_vertex_handles(self.selected_item)
        else:
            QMessageBox.warning(self, "警告", "多边形至少需要3个顶点")

    def create_vertex_handles(self, polygon_item):
        """创建多边形顶点手柄"""
        self.clear_vertex_handles()

        polygon = self.polygons[polygon_item.index]
        for i, point in enumerate(polygon['points']):
            # 创建顶点手柄
            handle = self.scene.addEllipse(
                point.x() - 5, point.y() - 5, 10, 10,
                QPen(QColor(255, 165, 0), 2),
                QBrush(QColor(255, 255, 255))
            )
            handle.setZValue(100)
            handle.setFlag(QGraphicsItem.ItemIsMovable, True)
            handle.setData(0, i)  # 存储顶点索引
            self.vertex_handles.append(handle)

    def clear_vertex_handles(self):
        """清除顶点手柄"""
        for handle in self.vertex_handles:
            self.scene.removeItem(handle)
        self.vertex_handles.clear()

    def highlight_vertex(self, vertex_index):
        """高亮显示指定顶点"""
        for i, handle in enumerate(self.vertex_handles):
            if i == vertex_index:
                handle.setPen(QPen(QColor(255, 0, 0), 3))
                handle.setBrush(QBrush(QColor(255, 200, 200)))
            else:
                handle.setPen(QPen(QColor(255, 165, 0), 2))
                handle.setBrush(QBrush(QColor(255, 255, 255)))

    def apply_polygon_edit(self):
        """应用多边形编辑"""
        if not self.selected_item or self.selected_item.annotation_type != 'polygon':
            return

        # 更新图形
        self.update_polygon_graphics()
        self.statusBar().showMessage('多边形已更新')

    def update_polygon_graphics(self):
        """更新多边形图形"""
        if not self.selected_item or self.selected_item.annotation_type != 'polygon':
            return

        # 更新图形项
        polygon = self.polygons[self.selected_item.index]
        self.selected_item.points = polygon['points']
        self.selected_item.polygon = QPolygonF(polygon['points'])
        self.selected_item.update()

    def add_calibration_point(self, pos):
        """添加校准点"""
        # 确保点击的是地图坐标，而不是场景坐标
        if self.map_item:
            map_pos = self.map_item.mapFromScene(pos)
            # 检查点是否在地图范围内
            if not self.map_item.boundingRect().contains(map_pos):
                self.statusBar().showMessage('请点击地图内的位置')
                return
            # 使用地图坐标
            self.calibration_points.append(map_pos)
        else:
            self.calibration_points.append(pos)

        # 显示标记
        pen = QPen(QColor(0, 255, 255), 3)
        brush = QBrush(QColor(0, 255, 255))
        marker = self.scene.addEllipse(
            QRectF(pos.x() - 5, pos.y() - 5, 10, 10),
            pen, brush
        )
        marker.setZValue(100)
        self.helper_items.append(marker)

        self.statusBar().showMessage(
            f'校准点 {len(self.calibration_points)}/2: ({pos.x():.1f}, {pos.y():.1f})'
        )

        if len(self.calibration_points) == 2:
            self.perform_calibration()
            self.calibration_points = []
            self.mode_buttons[EditMode.VIEW].setChecked(True)

    def perform_calibration(self):
        """执行两点校准 - 修复版本"""
        if len(self.circles) < 2:
            QMessageBox.warning(self, "警告", "需要至少两个圆形标注进行校准")
            return

        # 获取前两个圆形的当前位置
        circle1_item = None
        circle2_item = None

        for item in self.annotation_items:
            if isinstance(item, CircleAnnotation):
                if item.index == 0:
                    circle1_item = item
                elif item.index == 1:
                    circle2_item = item

        if not circle1_item or not circle2_item:
            QMessageBox.warning(self, "警告", "无法找到圆形标注")
            return

        # 获取圆形的场景坐标（考虑当前变换）
        circle1_scene = circle1_item.mapToScene(circle1_item.center)
        circle2_scene = circle2_item.mapToScene(circle2_item.center)

        # 计算当前向量
        current_vec = circle2_scene - circle1_scene
        current_dist = math.sqrt(current_vec.x() ** 2 + current_vec.y() ** 2)
        current_angle = math.atan2(current_vec.y(), current_vec.x())

        # 将校准点转换为场景坐标
        if self.map_item:
            calib_pt1_scene = self.map_item.mapToScene(self.calibration_points[0])
            calib_pt2_scene = self.map_item.mapToScene(self.calibration_points[1])
        else:
            calib_pt1_scene = self.calibration_points[0]
            calib_pt2_scene = self.calibration_points[1]

        # 计算目标向量
        target_vec = calib_pt2_scene - calib_pt1_scene
        target_dist = math.sqrt(target_vec.x() ** 2 + target_vec.y() ** 2)
        target_angle = math.atan2(target_vec.y(), target_vec.x())

        # 计算变换参数
        if current_dist > 0:
            scale = target_dist / current_dist
        else:
            scale = 1.0

        rotation = math.degrees(target_angle - current_angle)

        # 计算旋转和缩放后的第一个圆心位置
        # 首先获取所有标注的原始中心
        all_bounds = self.get_annotations_bounds()
        annotations_center = all_bounds.center()

        # 构建变换矩阵（先旋转缩放）
        transform = QTransform()
        transform.translate(annotations_center.x(), annotations_center.y())
        transform.rotate(rotation)
        transform.scale(scale, scale)
        transform.translate(-annotations_center.x(), -annotations_center.y())

        # 计算变换后第一个圆的位置
        transformed_circle1 = transform.map(circle1_scene)

        # 计算需要的平移量
        translation = calib_pt1_scene - transformed_circle1

        # 构建最终的变换矩阵
        final_transform = QTransform()
        final_transform.translate(translation.x(), translation.y())
        final_transform = final_transform * transform

        # 应用到所有标注
        for item in self.annotation_items:
            item.setTransform(final_transform)

        # 立即固化变换
        self.apply_current_transform()

        # 重置UI控件
        self.translate_x_spin.setValue(0)
        self.translate_y_spin.setValue(0)
        self.scale_spin.setValue(1.0)
        self.rotate_spin.setValue(0)

        self.statusBar().showMessage('校准完成')

    def clear_selection(self):
        """清除选择"""
        self.scene.clearSelection()
        self.selected_item = None
        self.delete_btn.setEnabled(False)
        self.instance_info_label.setText("未选中实例")
        self.circle_editor.hide()
        self.polygon_editor.hide()
        self.instance_mode_widget.hide()
        self.instance_transform_widget.hide()
        self.clear_vertex_handles()  # 清除顶点手柄

    def apply_circle_edit(self):
        """应用圆形编辑"""
        if not self.selected_item or self.selected_item.annotation_type != 'circle':
            return

        index = self.selected_item.index

        # 更新数据
        new_center = QPointF(self.circle_x_input.value(), self.circle_y_input.value())
        new_radius = self.circle_radius_input.value()

        self.circles[index]['center'] = new_center
        self.circles[index]['radius'] = new_radius

        # 更新图形项
        self.selected_item.center = new_center
        self.selected_item.radius = new_radius
        self.selected_item.update()

        self.statusBar().showMessage('圆形已更新')

    def delete_selected(self):
        """删除选中的实例"""
        if not self.selected_item:
            return

        reply = QMessageBox.question(
            self, '确认删除',
            f'确定要删除这个{self.selected_item.annotation_type}吗？',
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 从场景移除
            self.scene.removeItem(self.selected_item)
            self.annotation_items.remove(self.selected_item)

            # 从数据中移除
            if self.selected_item.annotation_type == 'circle':
                del self.circles[self.selected_item.index]
                # 更新索引
                for item in self.annotation_items:
                    if item.annotation_type == 'circle' and item.index > self.selected_item.index:
                        item.index -= 1
            elif self.selected_item.annotation_type == 'polygon':
                del self.polygons[self.selected_item.index]
                # 更新索引
                for item in self.annotation_items:
                    if item.annotation_type == 'polygon' and item.index > self.selected_item.index:
                        item.index -= 1

            self.clear_selection()
            self.statusBar().showMessage('实例已删除')

    def apply_transform(self):
        """应用变换参数"""
        # 获取变换参数
        translate = QPointF(self.translate_x_spin.value(), self.translate_y_spin.value())
        scale = self.scale_spin.value()
        rotation = self.rotate_spin.value()

        # 获取中心点
        if not hasattr(self, 'fixed_center'):
            # 计算所有原始标注的中心
            all_points = []

            # 添加所有原始坐标点
            for pos in self.trajectory_positions:
                all_points.append(pos)

            for circle in self.circles:
                center = circle['center']
                all_points.append(center)

            for poly in self.polygons:
                all_points.extend(poly['points'])

            if all_points:
                min_x = min(p.x() for p in all_points)
                max_x = max(p.x() for p in all_points)
                min_y = min(p.y() for p in all_points)
                max_y = max(p.y() for p in all_points)
                self.fixed_center = QPointF((min_x + max_x) / 2, (min_y + max_y) / 2)
            else:
                self.fixed_center = QPointF(0, 0)

        center = self.fixed_center

        # 构建统一的变换矩阵
        transform = QTransform()

        # 步骤1: 平移到原点
        transform.translate(-center.x(), -center.y())

        # 步骤2: 缩放
        transform.scale(scale, scale)

        # 步骤3: 旋转
        transform.rotate(rotation)

        # 步骤4: 平移回去并加上用户指定的平移
        transform.translate(center.x() + translate.x(), center.y() + translate.y())

        # 应用到所有标注
        for item in self.annotation_items:
            item.setPos(0, 0)  # 确保位置为0
            item.setTransform(transform)  # 应用统一的变换

    def apply_current_transform(self):
        """将当前的图形变换应用到实际数据"""
        if not self.annotation_items:
            return

        # 获取第一个标注项的变换
        transform = self.annotation_items[0].transform()

        # 计算实际的缩放因子
        unit_vector = QPointF(1.0, 0.0)
        transformed_vector = transform.map(unit_vector) - transform.map(QPointF(0, 0))
        scale_factor = math.sqrt(transformed_vector.x() ** 2 + transformed_vector.y() ** 2)

        # 检查是否有显著的缩放
        has_scaling = abs(scale_factor - 1.0) > 0.001

        # 更新轨迹数据
        for item in self.annotation_items:
            if isinstance(item, TrajectoryAnnotation):
                # 变换所有轨迹点
                new_positions = []
                for pos in item.positions:
                    new_pos = transform.map(pos)
                    new_positions.append(new_pos)

                # 更新轨迹数据
                self.trajectory_positions = new_positions

                # 重建路径
                item.positions = new_positions
                item.path = QPainterPath()
                item.path.moveTo(new_positions[0])
                for pos in new_positions[1:]:
                    item.path.lineTo(pos)

                # 重置变换
                item.setPos(0, 0)
                item.setTransform(QTransform())

        # 更新圆形数据
        for i, circle in enumerate(self.circles):
            item = next((item for item in self.annotation_items
                         if isinstance(item, CircleAnnotation) and item.index == i), None)
            if item:
                # 变换中心点
                new_center = transform.map(item.center)

                circle['center'] = new_center
                circle['data']['center'] = [new_center.x(), new_center.y()]

                # 如果变换包含缩放，更新半径
                if has_scaling:
                    if 0.1 < scale_factor < 10.0:
                        new_radius = item.radius * scale_factor
                        circle['radius'] = new_radius
                        circle['data']['radius'] = new_radius
                        # 更新图形项的半径
                        item.radius = new_radius

                # 更新图形项
                item.center = new_center
                item.setPos(0, 0)
                item.setTransform(QTransform())

        # 更新多边形数据
        for i, poly in enumerate(self.polygons):
            item = next((item for item in self.annotation_items
                         if isinstance(item, PolygonAnnotation) and item.index == i), None)
            if item:
                # 变换所有点
                new_points = [transform.map(p) for p in item.points]
                poly['points'] = new_points
                poly['data']['points'] = [[p.x(), p.y()] for p in new_points]

                # 更新图形项
                item.points = new_points
                item.polygon = QPolygonF(new_points)
                item.setPos(0, 0)
                item.setTransform(QTransform())

        # 重置变换参数
        self.translate_x_spin.setValue(0)
        self.translate_y_spin.setValue(0)
        self.scale_spin.setValue(1.0)
        self.rotate_spin.setValue(0)

    def reset_transform(self):
        """重置变换"""
        # 如果有未应用的变换，提醒用户
        if any(item.transform() != QTransform() for item in self.annotation_items):
            reply = QMessageBox.question(
                self, '重置变换',
                '当前有未应用的变换。是否要先应用这些变换？',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )

            if reply == QMessageBox.Yes:
                self.apply_current_transform()
            elif reply == QMessageBox.Cancel:
                return

        self.translate_x_spin.setValue(0)
        self.translate_y_spin.setValue(0)
        self.scale_spin.setValue(1.0)
        self.rotate_spin.setValue(0)

        for item in self.annotation_items:
            item.setTransform(QTransform())

    def center_annotations(self):
        """居中显示标注"""
        if not self.annotation_items:
            return

        # 获取标注边界
        bounds = self.get_annotations_bounds()

        # 计算需要的平移
        map_rect = self.map_item.boundingRect() if self.map_item else self.scene.sceneRect()
        map_center = map_rect.center()
        annotation_center = bounds.center()

        translation = map_center - annotation_center

        # 应用平移
        self.translate_x_spin.setValue(translation.x())
        self.translate_y_spin.setValue(translation.y())
        self.apply_transform()

        # 调整视图
        self.view.fitInView(bounds, Qt.KeepAspectRatio)

    def get_annotations_bounds(self):
        """获取所有标注的边界"""
        if not self.annotation_items:
            return QRectF()

        bounds = self.annotation_items[0].mapRectToScene(
            self.annotation_items[0].boundingRect()
        )

        for item in self.annotation_items[1:]:
            item_bounds = item.mapRectToScene(item.boundingRect())
            bounds = bounds.united(item_bounds)

        return bounds

    def reset_view(self):
        """重置视图"""
        self.view.reset_view()
        self.update_view_info()

    def update_view_info(self):
        """更新视图信息显示"""
        zoom = int(self.view.zoom_level * 100)
        rotation = int(self.view.rotation)

        self.zoom_label.setText(f"缩放: {zoom}%")
        self.rotation_label.setText(f"旋转: {rotation}°")

    def save_annotations(self):
        """保存标注数据"""
        if not self.tracking_data:
            QMessageBox.warning(self, "警告", "没有加载数据")
            return

        try:
            # 确保变换已应用
            if any(item.transform() != QTransform() for item in self.annotation_items):
                self.apply_current_transform()

            # 更新轨迹数据
            self.tracking_data['trajectory_summary']['trajectory_positions'] = [
                [pos.x(), pos.y()] for pos in self.trajectory_positions
            ]

            # 更新tracking_data中的圆形
            self.tracking_data['annotations']['circles'] = []
            for circle in self.circles:
                self.tracking_data['annotations']['circles'].append({
                    'center': [circle['center'].x(), circle['center'].y()],
                    'radius': circle['radius'],
                    'global_coords': True
                })

            # 更新tracking_data中的多边形
            self.tracking_data['annotations']['polygons'] = []
            for poly in self.polygons:
                points = [[p.x(), p.y()] for p in poly['points']]
                self.tracking_data['annotations']['polygons'].append({
                    'points': points,
                    'global_coords': True
                })

            # 保存文件
            with open(self.tracking_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_data, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f'已保存到：{self.tracking_data_path}')
            QMessageBox.information(self, "成功", "标注数据已保存")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    def export_annotated_image(self):
        """导出标注图片"""
        if not self.map_item:
            QMessageBox.warning(self, "警告", "没有加载地图")
            return

        # 确保变换已应用
        if any(item.transform() != QTransform() for item in self.annotation_items):
            self.apply_current_transform()

        # 获取保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"annotated_map_{timestamp}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存标注图片", default_name, "PNG图片 (*.png);;JPEG图片 (*.jpg)"
        )

        if not file_path:
            return

        try:
            # 创建图像（只包含地图区域）
            map_rect = self.map_item.boundingRect()
            image = QImage(map_rect.size().toSize(), QImage.Format_ARGB32)
            image.fill(Qt.transparent)

            # 渲染场景到图像
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            self.scene.render(painter, QRectF(image.rect()), map_rect)
            painter.end()

            # 保存图像
            image.save(file_path)
            self.statusBar().showMessage(f'图片已导出：{file_path}')
            QMessageBox.information(self, "成功", f"标注图片已保存到：\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 视图导航快捷键
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.view.scale(1.2, 1.2)
            self.view.zoom_level *= 1.2
            self.update_view_info()
        elif event.key() == Qt.Key_Minus:
            self.view.scale(0.8, 0.8)
            self.view.zoom_level *= 0.8
            self.update_view_info()
        elif event.key() == Qt.Key_0:
            self.view.reset_view()

        super().keyPressEvent(event)


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 创建并显示主窗口
    window = GlobalMapAnnotatorGUI()
    window.show()

    # 如果有命令行参数，自动加载
    if len(sys.argv) >= 3:
        window.data_path_input.setText(sys.argv[1])
        window.map_path_input.setText(sys.argv[2])
        window.load_data()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()