import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.is_playing = False
        self.speed_segments = []  # 存储加速段信息 [(start, end, speed)]

        self.initUI()

    def initUI(self):
        self.setWindowTitle('视频播放器 - 高级功能')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 8px;
                background: #3c3c3c;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #4CAF50;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部：文件选择区域
        file_widget = QWidget()
        file_layout = QHBoxLayout(file_widget)
        file_layout.addWidget(QLabel("视频路径:"))
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("请选择或输入视频文件路径...")
        file_layout.addWidget(self.path_input)
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)
        self.load_btn = QPushButton("加载视频")
        self.load_btn.clicked.connect(self.load_video)
        file_layout.addWidget(self.load_btn)
        main_layout.addWidget(file_widget)

        # 中间：视频显示和控制区域
        content_layout = QHBoxLayout()

        # 左侧：视频显示
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        # 播放控制
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)

        self.frame_label = QLabel("帧: 0/0")
        control_layout.addWidget(self.frame_label)
        control_layout.addStretch()
        video_layout.addWidget(control_widget)

        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.valueChanged.connect(self.slider_changed)
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_released)
        self.progress_slider.setEnabled(False)
        video_layout.addWidget(self.progress_slider)

        content_layout.addWidget(video_widget)

        # 右侧：功能控制面板
        control_panel = QWidget()
        control_panel.setMaximumWidth(350)
        control_panel_layout = QVBoxLayout(control_panel)

        # 1. 帧跳转功能
        jump_group = QGroupBox("帧跳转")
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("跳转到帧:"))
        self.jump_frame_input = QSpinBox()
        self.jump_frame_input.setMinimum(0)
        jump_layout.addWidget(self.jump_frame_input)
        self.jump_btn = QPushButton("跳转")
        self.jump_btn.clicked.connect(self.jump_to_frame)
        self.jump_btn.setEnabled(False)
        jump_layout.addWidget(self.jump_btn)
        jump_group.setLayout(jump_layout)
        control_panel_layout.addWidget(jump_group)

        # 2. 速度调整功能
        speed_group = QGroupBox("速度调整")
        speed_layout = QGridLayout()
        speed_layout.addWidget(QLabel("起始帧:"), 0, 0)
        self.speed_start_input = QSpinBox()
        self.speed_start_input.setMinimum(0)
        speed_layout.addWidget(self.speed_start_input, 0, 1)
        speed_layout.addWidget(QLabel("结束帧:"), 1, 0)
        self.speed_end_input = QSpinBox()
        self.speed_end_input.setMinimum(0)
        speed_layout.addWidget(self.speed_end_input, 1, 1)
        speed_layout.addWidget(QLabel("倍速:"), 2, 0)
        self.speed_rate_input = QDoubleSpinBox()
        self.speed_rate_input.setMinimum(0.1)
        self.speed_rate_input.setMaximum(10.0)
        self.speed_rate_input.setSingleStep(0.1)
        self.speed_rate_input.setValue(1.5)
        speed_layout.addWidget(self.speed_rate_input, 2, 1)
        self.apply_speed_btn = QPushButton("应用速度调整")
        self.apply_speed_btn.clicked.connect(self.apply_speed_adjustment)
        self.apply_speed_btn.setEnabled(False)
        speed_layout.addWidget(self.apply_speed_btn, 3, 0, 1, 2)
        self.export_video_btn = QPushButton("导出调速视频")
        self.export_video_btn.clicked.connect(self.export_video)
        self.export_video_btn.setEnabled(False)
        speed_layout.addWidget(self.export_video_btn, 4, 0, 1, 2)
        speed_group.setLayout(speed_layout)
        control_panel_layout.addWidget(speed_group)

        # 3. 帧导出功能
        export_group = QGroupBox("帧导出")
        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("导出帧:"))
        self.export_frame_input = QSpinBox()
        self.export_frame_input.setMinimum(0)
        export_layout.addWidget(self.export_frame_input)
        self.export_frame_btn = QPushButton("导出图片")
        self.export_frame_btn.clicked.connect(self.export_frame)
        self.export_frame_btn.setEnabled(False)
        export_layout.addWidget(self.export_frame_btn)
        export_group.setLayout(export_layout)
        control_panel_layout.addWidget(export_group)

        # 速度调整列表
        self.speed_list_group = QGroupBox("已应用的速度调整")
        speed_list_layout = QVBoxLayout()
        self.speed_list = QListWidget()
        self.speed_list.setMaximumHeight(150)
        speed_list_layout.addWidget(self.speed_list)
        self.clear_speeds_btn = QPushButton("清除所有调速")
        self.clear_speeds_btn.clicked.connect(self.clear_speed_adjustments)
        self.clear_speeds_btn.setEnabled(False)
        speed_list_layout.addWidget(self.clear_speeds_btn)
        self.speed_list_group.setLayout(speed_list_layout)
        control_panel_layout.addWidget(self.speed_list_group)

        control_panel_layout.addStretch()
        content_layout.addWidget(control_panel)

        main_layout.addLayout(content_layout)

        # 状态栏
        self.statusBar().showMessage('准备就绪')
        self.statusBar().setStyleSheet("color: white; background-color: #1e1e1e;")

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "",
                                                   "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv)")
        if file_path:
            self.path_input.setText(file_path)

    def load_video(self):
        video_path = self.path_input.text()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "警告", "请选择有效的视频文件路径")
            return

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.error(self, "错误", "无法打开视频文件")
            return

        self.video_path = video_path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        self.speed_segments = []

        # 更新UI
        self.progress_slider.setMaximum(self.total_frames - 1)
        self.progress_slider.setValue(0)
        self.jump_frame_input.setMaximum(self.total_frames - 1)
        self.speed_start_input.setMaximum(self.total_frames - 1)
        self.speed_end_input.setMaximum(self.total_frames - 1)
        self.export_frame_input.setMaximum(self.total_frames - 1)

        # 启用控件
        self.play_btn.setEnabled(True)
        self.progress_slider.setEnabled(True)
        self.jump_btn.setEnabled(True)
        self.apply_speed_btn.setEnabled(True)
        self.export_video_btn.setEnabled(True)
        self.export_frame_btn.setEnabled(True)
        self.clear_speeds_btn.setEnabled(True)

        # 显示第一帧
        self.show_frame(0)
        self.statusBar().showMessage(
            f'已加载视频: {os.path.basename(video_path)} | 总帧数: {self.total_frames} | FPS: {self.fps:.2f}')

    def show_frame(self, frame_idx):
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            # 转换颜色空间并显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # 保持宽高比缩放
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

            self.current_frame = frame_idx
            self.progress_slider.setValue(frame_idx)
            self.frame_label.setText(f"帧: {frame_idx}/{self.total_frames - 1}")

    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("播放")
            self.is_playing = False
        else:
            # 计算定时器间隔
            base_interval = int(1000 / self.fps)
            self.timer.start(base_interval)
            self.play_btn.setText("暂停")
            self.is_playing = True

    def play_video(self):
        if self.current_frame < self.total_frames - 1:
            # 检查当前帧是否在加速段中
            speed_factor = 1.0
            for start, end, speed in self.speed_segments:
                if start <= self.current_frame < end:
                    speed_factor = speed
                    break

            # 根据速度因子跳帧
            next_frame = min(self.current_frame + int(speed_factor), self.total_frames - 1)
            self.show_frame(next_frame)
        else:
            self.toggle_play()
            self.current_frame = 0
            self.show_frame(0)

    def slider_pressed(self):
        if self.is_playing:
            self.timer.stop()

    def slider_released(self):
        if self.is_playing:
            self.timer.start(int(1000 / self.fps))

    def slider_changed(self, value):
        if not self.is_playing:
            self.show_frame(value)

    def jump_to_frame(self):
        frame_idx = self.jump_frame_input.value()
        self.show_frame(frame_idx)

    def apply_speed_adjustment(self):
        start = self.speed_start_input.value()
        end = self.speed_end_input.value()
        speed = self.speed_rate_input.value()

        if start >= end:
            QMessageBox.warning(self, "警告", "起始帧必须小于结束帧")
            return

        # 添加到速度调整列表
        self.speed_segments.append((start, end, speed))
        self.speed_list.addItem(f"帧 {start}-{end}: {speed}x")
        self.statusBar().showMessage(f"已应用速度调整: 帧 {start}-{end} 设置为 {speed}x")

    def clear_speed_adjustments(self):
        self.speed_segments = []
        self.speed_list.clear()
        self.statusBar().showMessage("已清除所有速度调整")

    def export_video(self):
        if not self.cap or not self.speed_segments:
            QMessageBox.warning(self, "警告", "请先加载视频并应用速度调整")
            return

        # 生成输出文件名
        base_name = os.path.splitext(self.video_path)[0]
        output_path = f"{base_name}_speed_adjusted.mp4"

        # 获取视频参数
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        # 创建进度对话框
        progress = QProgressDialog("正在导出视频...", "取消", 0, self.total_frames, self)
        progress.setWindowModality(Qt.WindowModal)

        # 处理每一帧
        frame_idx = 0
        while frame_idx < self.total_frames:
            if progress.wasCanceled():
                break

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                break

            # 检查是否在加速段
            speed_factor = 1.0
            for start, end, speed in self.speed_segments:
                if start <= frame_idx < end:
                    speed_factor = speed
                    break

            # 如果不是加速段或减速段，正常写入
            if speed_factor == 1.0:
                out.write(frame)
                frame_idx += 1
            elif speed_factor > 1.0:
                # 加速：跳帧
                out.write(frame)
                frame_idx += int(speed_factor)
            else:
                # 减速：重复帧
                repeat_times = int(1 / speed_factor)
                for _ in range(repeat_times):
                    out.write(frame)
                frame_idx += 1

            progress.setValue(frame_idx)

        out.release()
        progress.close()

        if not progress.wasCanceled():
            QMessageBox.information(self, "成功", f"视频已导出到:\n{output_path}")
            self.statusBar().showMessage(f"视频已导出: {output_path}")

    def export_frame(self):
        frame_idx = self.export_frame_input.value()

        if not self.cap:
            QMessageBox.warning(self, "警告", "请先加载视频")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            # 生成输出文件名
            base_name = os.path.splitext(self.video_path)[0]
            output_path = f"{base_name}_frame_{frame_idx}.png"

            # 保存图片
            cv2.imwrite(output_path, frame)
            QMessageBox.information(self, "成功", f"帧 {frame_idx} 已保存到:\n{output_path}")
            self.statusBar().showMessage(f"已导出帧 {frame_idx}: {output_path}")
        else:
            QMessageBox.error(self, "错误", "无法读取指定帧")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()