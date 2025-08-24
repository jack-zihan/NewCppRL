import pandas as pd
from pyproj import Transformer, CRS
import json

def latlon_to_utm(lat, lon):
    # 计算 UTM 分区号
    utm_zone_number = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'

    # 根据 UTM 分区号和半球确定 EPSG 代码
    if hemisphere == 'north':
        epsg_code = 32600 + utm_zone_number  # WGS84 北半球
    else:
        epsg_code = 32700 + utm_zone_number  # WGS84 南半球

    # 定义 UTM 坐标系
    utm_crs = CRS.from_epsg(epsg_code)

    # 定义 WGS84 地理坐标系
    wgs84_crs = CRS.from_epsg(4326)

    # 创建坐标转换器
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

    # 执行转换（经度，纬度）
    easting, northing = transformer.transform(lon, lat)

    return {
        "easting": easting,
        "northing": northing,
        "zone_number": utm_zone_number,
        "hemisphere": hemisphere
    }

def process_csv_file(csv_file_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path, header=None)

    # 提取经度和纬度（第三列和第四列，索引为 2 和 3）
    lons = df.iloc[:, 2]  # 第三列：经度
    lats = df.iloc[:, 3]  # 第四列：纬度

    # 存储转换后的 UTM 坐标点
    utm_points = []

    for idx, (lat, lon) in enumerate(zip(lats, lons)):
        # 检查数据是否为有效数字
        if pd.notnull(lat) and pd.notnull(lon):
            try:
                utm_coord = latlon_to_utm(lat, lon)
                utm_points.append([utm_coord["easting"], utm_coord["northing"]])
            except Exception as e:
                print(f"第 {idx + 1} 行数据转换出错：{e}")
        else:
            print(f"跳过无效数据点（第 {idx + 1} 行）：纬度={lat}, 经度={lon}")

    return utm_points

if __name__ == "__main__":
    # 输入 CSV 文件路径
    frontier_csv = 'frontier.csv'
    obstacle_csv = 'obstacle.csv'
    weed_csv = 'weeds.csv'

    # 处理地图边界点
    map_frontier_edge_points = process_csv_file(frontier_csv)

    # 处理障碍物边界点
    # 假设 obstacle_csv 文件中，每个障碍物之间用空行分隔
    with open(obstacle_csv, 'r') as f:
        content = f.read()

    obstacle_blocks = content.strip().split('\n\n')
    map_obstacle_edge_points = []
    for block_idx, block in enumerate(obstacle_blocks):
        # 将每个障碍物的数据解析为 DataFrame
        lines = block.strip().split('\n')
        data = [line.split(',') for line in lines]
        temp_df = pd.DataFrame(data)
        temp_df = temp_df.apply(pd.to_numeric, errors='coerce')

        # 提取经度和纬度
        lons = temp_df.iloc[:, 2]  # 第三列：经度
        lats = temp_df.iloc[:, 3]  # 第四列：纬度

        utm_points = []
        for idx, (lat, lon) in enumerate(zip(lats, lons)):
            if pd.notnull(lat) and pd.notnull(lon):
                try:
                    utm_coord = latlon_to_utm(lat, lon)
                    utm_points.append([utm_coord["easting"], utm_coord["northing"]])
                except Exception as e:
                    print(f"障碍物 {block_idx + 1}，第 {idx + 1} 行数据转换出错：{e}")
            else:
                print(f"跳过无效数据点：纬度={lat}, 经度={lon}")

        map_obstacle_edge_points.append(utm_points)

    # 处理杂草点
    map_weed_points = process_csv_file(weed_csv)

    # 准备输出的 JSON 数据
    output_data = {
        "map_frontier_edge_points": map_frontier_edge_points,
        "map_obstacle_edge_points": map_obstacle_edge_points,
        "map_weed_points": map_weed_points
    }

    # 将数据写入 JSON 文件
    json_output_path = "your_output_file.json"   # 请替换为你想要的输出 JSON 文件路径

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"UTM 坐标已保存到 {json_output_path}")
