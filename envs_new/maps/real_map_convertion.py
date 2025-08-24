from pathlib import Path
import numpy as np
import cv2
import os
import json

# from rl.sac_cont.sac_cont_reaLtest import real_map_dir


def generate_maps(
    map_frontier_edge_points: np.ndarray,
    map_obstacle_edge_points: list[np.ndarray],
    map_weed_points: np.ndarray,
    image_size: tuple[int, int],
    save_dir: str
):

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create blank images
    width, height = image_size
    map_frontier = np.zeros((height, width), dtype=np.uint8)
    map_obstacle = np.zeros((height, width), dtype=np.uint8)
    map_weed = np.zeros((height, width), dtype=np.uint8)

    # Calculate centroid of the map frontier to center it
    frontier_centroid = np.mean(map_frontier_edge_points, axis=0)
    image_center = np.array([width / 2, height / 2])
    translation = image_center - frontier_centroid
    # Save the translation to a JSON file
    translation_dict = {'translation': translation.tolist()}
    with open(os.path.join(save_dir, 'translation.json'), 'w') as f:
        json.dump(translation_dict, f)

    # Function to apply translation to points
    def translate_points(points, translation):
        return points + translation

    # Translate map frontier points (points are in (x, y) order)
    map_frontier_points_translated = translate_points(map_frontier_edge_points, translation).astype(np.int32)

    # Draw map frontier
    cv2.fillPoly(
        map_frontier,
        [map_frontier_points_translated],
        color=255  # White color in grayscale
    )

    # Translate and draw obstacles
    for obstacle_points in map_obstacle_edge_points:
        obstacle_points_translated = translate_points(obstacle_points, translation).astype(np.int32)
        cv2.fillPoly(
            map_obstacle,
            [obstacle_points_translated],
            color=255
        )
    if len(map_weed_points)!=0:
        # Translate and plot weed points
        map_weed_points_translated = translate_points(map_weed_points, translation).astype(np.int32)
        for x, y in map_weed_points_translated:
            # Check bounds to prevent indexing errors
            if 0 <= x < width and 0 <= y < height:
                map_weed[y, x] = 255  # Set pixel at (y, x)

    # Save images
    cv2.imwrite(os.path.join(save_dir, 'map_field.png'), map_frontier)
    cv2.imwrite(os.path.join(save_dir, 'map_obstacle.png'), map_obstacle)
    cv2.imwrite(os.path.join(save_dir, 'map_weed.png'), map_weed)

    print(f"Maps saved to {save_dir}")
    print(f"Translation saved to {os.path.join(save_dir, 'translation.json')}")

def restore_coordinates(points: np.ndarray, json_path: str) -> np.ndarray:
    # Load the translation from the JSON file
    with open(json_path, 'r') as f:
        translation_dict = json.load(f)
    translation = np.array(translation_dict['translation'])

    # Apply inverse translation
    restored_points = points - translation
    return restored_points

def load_map_data(real_map_point_dir: str):
    # Load map data from a JSON file
    with open(real_map_point_dir, 'r') as f:
        data = json.load(f)

    # Extract the frontier, obstacle, and weed points from the JSON
    map_frontier_edge_points = np.array(data['map_frontier_edge_points'])
    map_obstacle_edge_points = [np.array(points) for points in data['map_obstacle_edge_points']]
    map_weed_points = np.array(data['map_weed_points'])

    return map_frontier_edge_points, map_obstacle_edge_points, map_weed_points


# Example usage remains the same
if __name__ == "__main__":
    real_map_point_dir = "/home/lzh/NewCppRL/envs/maps/real_true/your_output_file.json"
    map_frontier_edge_points, map_obstacle_edge_points, map_weed_points = load_map_data(real_map_point_dir)
    # map_frontier_edge_points = np.array([
    #     [421644.1808135206, 4311531.640420083],
    #     [421544.2941347627, 4311535.406707985],
    #     [421539.8297026052, 4311426.242815197],
    #     [421638.78655665725, 4311424.705410738]
    # ])  # A square frontier
    #
    # # map_frontier_edge_points = np.array([
    # #     [646394.8130324452, 4130893.0518099137],
    # #     [646394.894777249, 4130878.525252316],
    # #     [646401.6436589845, 4130877.8552505348],
    # #     [646401.0156579893, 4130893.493210295]
    # # ])  # A square frontier
    #
    # map_obstacle_edge_points = [
    #     # np.array([
    #     #     [150, 150],
    #     #     [200, 150],
    #     #     [200, 200],
    #     #     [150, 200]
    #     # ]),  # First obstacle
    #     # np.array([
    #     #     [220, 220],
    #     #     [260, 220],
    #     #     [260, 260],
    #     #     [220, 260]
    #     # ])   # Second obstacle
    # ]
    #
    # map_weed_points = np.array([
    #     # [120, 120],
    #     # [280, 280],
    #     # [160, 240]
    # ])

    # Define image size and save directory
    image_size = (400, 400)
    base_dir = Path(__file__).parent
    save_dir = f'{base_dir}/real_true'
    # save_dir = f'{base_dir}/real_test_1'
    print(save_dir)
    # # Generate and save maps
    generate_maps(
        map_frontier_edge_points,
        map_obstacle_edge_points,
        map_weed_points,
        image_size,
        save_dir
    )