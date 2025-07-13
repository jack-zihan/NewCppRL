from pathlib import Path
import numpy as np
import cv2
import os
import json

def echo_boudingbox(points, name="Unnamed points"):
    """
    Calculate and print bounding box information for a set of points
    
    Args:
        points: Point set (numpy array or list of numpy arrays)
        name: Name of the point set for display
    """
    if isinstance(points, list):
        # Handle list of point sets (e.g., map_obstacle_edge_points)
        total_area = 0
        print(f"{name}:")
        for i, point_set in enumerate(points):
            if point_set is None or not isinstance(point_set, np.ndarray) or len(point_set) < 3:
                print(f"  Set {i+1}: Not enough points to form a rectangle")
                continue

            # Ensure points are in the right format for OpenCV
            pts = np.array(point_set, dtype=np.float32)
            if pts.size == 0 or pts.ndim != 2 or pts.shape[1] != 2:
                print(f"  Set {i+1}: Invalid point format")
                continue

            try:
                rect = cv2.minAreaRect(pts)
                width, height = rect[1]
                area = width * height
                total_area += area
                print(f"  Set {i+1}: Width = {width:.2f}m, Height = {height:.2f}m, Area = {area:.2f}m²")
            except Exception as e:
                print(f"  Set {i+1}: Error calculating bounding box - {str(e)}")

        if total_area > 0:
            print(f"  Total Area: {total_area:.2f}m²")
        else:
            print(f"  No valid obstacle areas found")
    else:
        # Handle single point set (e.g., map_frontier_edge_points or map_weed_points)
        if points is None or not isinstance(points, np.ndarray) or len(points) == 0:
            print(f"{name}: No points available")
            return

        if len(points) < 3:  # Need at least 3 points for a rectangle
            print(f"{name}: Not enough points to form a rectangle")
            return

        # Ensure points are in the right format for OpenCV
        pts = np.array(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            print(f"{name}: Invalid point format - expected shape (N,2), got {pts.shape}")
            return

        # For weed points (which may be scattered), calculate axis-aligned bounding box
        if "weed" in name.lower():
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            print(f"{name}: Width = {width:.2f}m, Height = {height:.2f}m, Area = {area:.2f}m²")
        else:
            # For other point sets, compute minimum area rectangle
            try:
                rect = cv2.minAreaRect(pts)
                width, height = rect[1]
                area = width * height
                print(f"{name}: Width = {width:.2f}m, Height = {height:.2f}m, Area = {area:.2f}m²")
            except Exception as e:
                print(f"{name}: Error calculating bounding box - {str(e)}")

def generate_maps(
    map_frontier_edge_points: np.ndarray,
    map_obstacle_edge_points: list[np.ndarray],
    map_weed_points: np.ndarray,
    image_size: tuple[int, int],
    save_dir: str,
    scale: float = 1.0  # Scale factor: meters per pixel
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

    # Compute the translation to center the map in the image
    image_center = np.array([width / 2, height / 2]) * scale  # Convert pixels to meters
    translation = image_center - frontier_centroid

    # Save the translation and scale to a JSON file
    transformation_info = {
        'translation': translation.tolist(),
        'scale': scale
    }
    with open(os.path.join(save_dir, 'transformation.json'), 'w') as f:
        json.dump(transformation_info, f, indent=4)

    # Function to apply translation and scaling to points
    def transform_points(points, translation, scale):
        # Apply translation and scaling: (world_coords + translation) / scale -> image_coords
        return ((points + translation) / scale).astype(np.int32)

    # Transform map frontier points
    map_frontier_points_transformed = transform_points(map_frontier_edge_points, translation, scale)

    # Draw map frontier
    cv2.fillPoly(
        map_frontier,
        [map_frontier_points_transformed],
        color=255  # White color in grayscale
    )

    # Transform and draw obstacles
    for obstacle_points in map_obstacle_edge_points:
        obstacle_points_transformed = transform_points(obstacle_points, translation, scale)
        cv2.fillPoly(
            map_obstacle,
            [obstacle_points_transformed],
            color=255
        )

    # Transform and plot weed points
    if len(map_weed_points) != 0:
        map_weed_points_transformed = transform_points(map_weed_points, translation, scale)
        for x, y in map_weed_points_transformed:
            # Check bounds to prevent indexing errors
            if 0 <= x < width and 0 <= y < height:
                map_weed[y, x] = 255  # Set pixel at (y, x)

    # Save images
    cv2.imwrite(os.path.join(save_dir, 'map_frontier.png'), map_frontier)
    cv2.imwrite(os.path.join(save_dir, 'map_obstacle.png'), map_obstacle)
    cv2.imwrite(os.path.join(save_dir, 'map_weed.png'), map_weed)

    print(f"Maps saved to {save_dir}")
    print(f"Transformation info saved to {os.path.join(save_dir, 'transformation.json')}")

def restore_coordinates(points: np.ndarray, json_path: str) -> np.ndarray:
    # Load the translation and scale from the JSON file
    with open(json_path, 'r') as f:
        transformation_info = json.load(f)
    translation = np.array(transformation_info['translation'])
    scale = transformation_info['scale']

    # Apply inverse scaling and translation: (image_coords * scale) - translation -> world_coords
    restored_points = (np.array(points) * scale) - translation
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

# Example usage
if __name__ == "__main__":
    real_map_point_dir = "/home/lzh/NewCppRL/envs/maps/real_true/your_output_file.json"
    map_frontier_edge_points, map_obstacle_edge_points, map_weed_points = load_map_data(real_map_point_dir)

    # Output bounding box information for all point sets
    print("\n=== Bounding Box Information ===")
    echo_boudingbox(map_frontier_edge_points, "Map frontier edge points")
    echo_boudingbox(map_obstacle_edge_points, "Map obstacle edge points")
    echo_boudingbox(map_weed_points, "Map weed points")
    print("===============================\n")

    # Define image size and save directory
    image_size = (400, 400)  # Width and height in pixels
    scale = 0.3  # Meters per pixel

    base_dir = Path(__file__).parent
    save_dir = base_dir / 'real_true'
    print(f"Saving maps to: {save_dir}")

    # Generate and save maps
    generate_maps(
        map_frontier_edge_points,
        map_obstacle_edge_points,
        map_weed_points,
        image_size,
        str(save_dir),
        scale=scale
    )

