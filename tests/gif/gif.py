from PIL import Image, ImageDraw

# 第一个 GIF：绿色矩形，宽度不断拉长（左边固定），无空白区域
def create_green_rectangle_gif():
    frames = []
    width, height = 50, 100  # 初始宽度和高度
    max_width = 300  # 最大宽度

    # 创建固定尺寸的图像
    img_size = (max_width, height)

    # 增加步长，使增长更快
    for w in range(width, max_width + 1, 15):
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        # 绘制绿色矩形，左边固定在 (0, 0)
        draw.rectangle([0, 0, w, height], fill='green')
        frames.append(img)

    # 保存 GIF，减少帧持续时间，使动画更快
    frames[0].save('green_rectangle.gif', save_all=True, append_images=frames[1:], optimize=False, duration=50, loop=0)

# 第二个 GIF：红色小矩形，每隔一段时间往右复制一个（有间隙），复制速度变慢
def create_red_blocks_gif():
    frames = []
    block_width, height = 20, 100  # 小矩形的宽度和高度
    gap = 5  # 间隙
    max_blocks = 10  # 最大矩形数量
    # 计算总宽度，固定图像尺寸
    total_width = max_blocks * (block_width + gap) - gap
    img_size = (total_width, height)

    for n in range(1, max_blocks + 1):
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        for i in range(n):
            x0 = i * (block_width + gap)
            x1 = x0 + block_width
            draw.rectangle([x0, 0, x1, height], fill='red')
        frames.append(img)

    # 保存 GIF，增加帧持续时间，使动画变慢
    frames[0].save('red_blocks.gif', save_all=True, append_images=frames[1:], optimize=False, duration=1000, loop=0)

if __name__ == '__main__':
    create_green_rectangle_gif()
    create_red_blocks_gif()
