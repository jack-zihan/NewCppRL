import cv2
import numpy as np

# 设置标识符
record_video = True  # 如果为 True，则保存处理后的视频
add_watermark = True  # 如果为 True，则在视频帧上添加水印

# 输入视频路径
video_path = 'real_world_video.mp4'  # 请替换为您的视频文件路径

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频帧率和大小
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 如果需要记录视频，初始化视频写入器
if record_video:
    output_video_path = 'add_traj.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 跳过前49帧（第0帧算起）
frame_count = 0
# while frame_count < 50:
#     ret, frame = cap.read()
#     if not ret:
#         print("无法读取视频帧")
#         exit()
#     frame_count += 1

# 读取第50帧
ret, first_frame = cap.read()
if not ret:
    print("无法读取视频帧")
    exit()

# 让用户在第50帧中选择目标
bbox = cv2.selectROI('Select Target', first_frame, False)
cv2.destroyWindow('Select Target')

# 初始化目标追踪器，使用CSRT追踪器以提高鲁棒性
tracker = cv2.TrackerCSRT_create()

# 初始化追踪器
tracker.init(first_frame, bbox)

# 用于稳像的参数
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 初始化累计变换矩阵为单位矩阵
cumulative_transform = np.eye(3, 3, dtype=np.float32)

# 初始化轨迹点列表
trajectory = []

# 设置平滑窗口大小
smoothing_window_size = 5

# 准备水印（如果需要）
if add_watermark:
    # 创建水印图像
    watermark_text = 'TIANLIANG'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size, _ = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)

    # 创建一个透明的水印图像，大小与帧大小相同
    watermark = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # 在水印图像上重复绘制水印文字
    for y in range(0, frame_height, text_size[1] * 3):
        for x in range(0, frame_width, text_size[0] * 3):
            cv2.putText(watermark, watermark_text, (x, y + text_size[1]), font, font_scale, (0, 0, 0), font_thickness,
                        cv2.LINE_AA)
            cv2.putText(watermark, watermark_text, (x + 1, y + text_size[1] + 1), font, font_scale, (0, 0, 0),
                        font_thickness, cv2.LINE_AA)
            cv2.putText(watermark, watermark_text, (x - 1, y + text_size[1] - 1), font, font_scale, (255, 255, 255),
                        font_thickness, cv2.LINE_AA)

    # 生成水印掩码
    watermark_mask = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
    _, watermark_mask = cv2.threshold(watermark_mask, 1, 255, cv2.THRESH_BINARY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 更新目标追踪器
    success, bbox = tracker.update(frame)
    if not success:
        print("目标丢失，停止追踪")
        break
    x, y, w_box, h_box = [int(v) for v in bbox]

    # 在当前帧绘制目标边界框
    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

    # 将当前帧转换为灰度图像
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用光流法估计相机运动
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # 过滤有效的点
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # 计算变换矩阵
    if len(prev_pts) >= 4 and len(curr_pts) >= 4:
        m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    else:
        m = None

    # 更新累计变换矩阵
    if m is not None:
        # 将仿射矩阵转换为3x3矩阵
        m_homo = np.vstack([m, [0, 0, 1]])  # 将 (2, 3) 转换为 (3, 3)
        cumulative_transform = cumulative_transform @ np.linalg.inv(m_homo)
    else:
        # 如果无法估计变换矩阵，累计变换矩阵不变
        pass

    # 更新前一帧和灰度图像
    prev_gray = curr_gray.copy()

    # 获取目标中心点的位置
    position = np.array([[x + w_box / 2, y + h_box / 2]], dtype=np.float32).reshape(-1, 1, 2)  # Shape: (1, 1, 2)

    # 将目标位置转换到初始帧的坐标系中
    position_in_initial = cv2.perspectiveTransform(position, cumulative_transform)

    # 提取转换后的坐标并存储为一维数组
    position_in_initial = position_in_initial[0, 0]  # Shape: (2,)

    # 将轨迹点添加到列表
    trajectory.append(position_in_initial)

    # 对轨迹点进行平滑处理
    trajectory_array = np.array(trajectory)
    if len(trajectory_array) >= smoothing_window_size:
        # 使用边界值填充，减少边缘效应
        pad_size = smoothing_window_size // 2
        x_padded = np.pad(trajectory_array[:, 0], (pad_size, pad_size), 'edge')
        y_padded = np.pad(trajectory_array[:, 1], (pad_size, pad_size), 'edge')

        kernel = np.ones(smoothing_window_size) / smoothing_window_size
        x_smooth = np.convolve(x_padded, kernel, mode='valid')
        y_smooth = np.convolve(y_padded, kernel, mode='valid')

        smoothed_trajectory = np.vstack((x_smooth, y_smooth)).T
    else:
        smoothed_trajectory = trajectory_array

    # 在当前帧上绘制轨迹
    if len(smoothed_trajectory) >= 2:
        # 将平滑后的轨迹点转换为形状 (N, 1, 2) 的数组
        smoothed_trajectory_points = smoothed_trajectory.reshape(-1, 1, 2).astype(np.float32)

        # 将轨迹点从初始帧坐标系转换回当前帧坐标系
        inv_cumulative_transform = np.linalg.inv(cumulative_transform)
        smoothed_trajectory_current = cv2.perspectiveTransform(smoothed_trajectory_points, inv_cumulative_transform)

        # 创建半透明的轨迹叠加层
        overlay = frame.copy()
        alpha = 1  # 轨迹透明度

        # 绘制轨迹
        for i in range(1, len(smoothed_trajectory_current)):
            pt1 = (int(smoothed_trajectory_current[i - 1][0][0]), int(smoothed_trajectory_current[i - 1][0][1]))
            pt2 = (int(smoothed_trajectory_current[i][0][0]), int(smoothed_trajectory_current[i][0][1]))
            cv2.line(overlay, pt1, pt2, (0, 0, 255), thickness=2)  # 增加线条宽度

        # 将轨迹叠加层与当前帧融合
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 添加水印（如果需要）
    if add_watermark:
        # 将水印叠加到当前帧
        frame = cv2.addWeighted(frame, 1, watermark, 0.3, 0)

    # 显示结果
    cv2.imshow('Trajectory', frame)

    # 如果需要记录视频，写入帧
    if record_video:
        out.write(frame)

    # 按键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放资源
cap.release()
if record_video:
    out.release()
cv2.destroyAllWindows()
