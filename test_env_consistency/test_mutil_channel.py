import cv2
import numpy as np
import sys


def test_multichannel_warpAffine():
    """测试 cv2.warpAffine 对不同通道数的 borderValue 支持"""

    print("=" * 60)
    print("测试 cv2.warpAffine borderValue 多通道支持")
    print(f"OpenCV 版本: {cv2.__version__}")
    print("=" * 60)

    # 创建一个简单的旋转矩阵（旋转45度）
    def get_rotation_matrix(size):
        center = (size // 2, size // 2)
        angle = 45
        scale = 0.7  # 缩小一点以便看到边界
        return cv2.getRotationMatrix2D(center, angle, scale)

    # 测试不同通道数
    test_configs = [
        # (1, (128,), "灰度图像 (1通道)"),
        # (3, (255, 128, 0), "RGB图像 (3通道)"),
        # (4, (255, 128, 0, 200), "RGBA图像 (4通道)"),
        (5, (255, 128, 0, 200, 100), "5通道图像"),
        (6, (255, 128, 0, 200, 100, 50), "6通道图像"),
    ]

    img_size = 50
    output_size = 100

    for n_channels, border_values, description in test_configs:
        print(f"\n测试 {description}")
        print("-" * 40)

        # 创建测试图像 - 中心有个白色方块
        img = np.zeros((img_size, img_size, n_channels), dtype=np.uint8)
        img[20:30, 20:30, :] = 255  # 中心白色方块

        M = get_rotation_matrix(img_size)

        # 测试1: 使用单个值
        try:
            single_value = 100
            result = cv2.warpAffine(img, M, (output_size, output_size),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=single_value)

            # 检查边界值
            corner_pixel = result[0, 0]
            print(f"✓ 单值填充 ({single_value}) 成功")
            print(f"  边角像素值: {corner_pixel}")

            # 检查是否所有通道都被填充
            if n_channels > 1:
                if np.all(corner_pixel == single_value):
                    print(f"  所有通道都填充为 {single_value}")
                else:
                    print(f"  警告: 只有部分通道被填充")
                    print(f"  期望: 所有通道={single_value}, 实际: {corner_pixel}")
        except Exception as e:
            print(f"✗ 单值填充失败: {str(e)[:100]}")

        # 测试2: 使用多通道值
        try:
            # 根据通道数调整 border_values
            if n_channels <= len(border_values):
                border_val_to_use = border_values[:n_channels]
            else:
                # 如果通道数大于提供的值，扩展值列表
                border_val_to_use = border_values

            result = cv2.warpAffine(img, M, (output_size, output_size),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=border_val_to_use)

            # 检查边界值
            corner_pixel = result[0, 0]
            print(f"✓ 多值填充 {border_val_to_use} 成功")
            print(f"  边角像素值: {corner_pixel}")

            # 验证每个通道是否正确填充
            if n_channels <= 4:
                expected = np.array(border_val_to_use[:n_channels])
                if np.array_equal(corner_pixel, expected):
                    print(f"  所有通道填充正确!")
                else:
                    print(f"  警告: 填充值不匹配")
                    print(f"  期望: {expected}, 实际: {corner_pixel}")

        except Exception as e:
            print(f"✗ 多值填充失败: {str(e)[:100]}")

        # 测试3: 特殊测试 - 对于>4通道，尝试只提供4个值
        if n_channels > 4:
            try:
                border_val_4 = border_values[:4]
                result = cv2.warpAffine(img, M, (output_size, output_size),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=border_val_4)
                corner_pixel = result[0, 0]
                print(f"✓ 使用4个值填充{n_channels}通道图像 成功")
                print(f"  使用的值: {border_val_4}")
                print(f"  边角像素值: {corner_pixel}")
                print(f"  注意: 后面的通道可能使用默认值")
            except Exception as e:
                print(f"✗ 使用4个值失败: {str(e)[:100]}")

    # 额外测试：使用 tuple vs list
    print("\n" + "=" * 60)
    print("额外测试: tuple vs list")
    print("-" * 40)

    img_3ch = np.zeros((50, 50, 3), dtype=np.uint8)
    img_3ch[20:30, 20:30, :] = 255
    M = get_rotation_matrix(50)

    # 测试 tuple
    try:
        border_tuple = (255, 128, 64)
        result = cv2.warpAffine(img_3ch, M, (100, 100),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_tuple)
        print(f"✓ Tuple {border_tuple} 成功")
        print(f"  边角像素: {result[0, 0]}")
    except Exception as e:
        print(f"✗ Tuple 失败: {str(e)}")

    # 测试 list
    try:
        border_list = [255, 128, 64]
        result = cv2.warpAffine(img_3ch, M, (100, 100),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_list)
        print(f"✓ List {border_list} 成功")
        print(f"  边角像素: {result[0, 0]}")
    except Exception as e:
        print(f"✗ List 失败: {str(e)}")

    # 测试 numpy array
    try:
        border_array = np.array([255, 128, 64])
        result = cv2.warpAffine(img_3ch, M, (100, 100),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_array)
        print(f"✓ NumPy array 成功")
        print(f"  边角像素: {result[0, 0]}")
    except Exception as e:
        print(f"✗ NumPy array 失败: {str(e)}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_multichannel_warpAffine()