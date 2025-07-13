import cv2
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class VideoMapStitcher:
    def __init__(self, video_path, frame_interval=30, max_frames=100, mode="scans"):
        """
        Initialize the video stitcher

        Args:
            video_path: Path to input video
            frame_interval: Extract one frame every N frames
            max_frames: Maximum number of frames to extract
            mode: Stitching mode ('panorama' or 'scans')
        """
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.mode = mode

        # Initialize containers
        self.frames = []
        self.frame_indices = []
        self.stitched_map = None
        self.frame_corners = []  # Store corners of each frame in the stitched map
        self.homographies = []  # Store transformation matrices

    def extract_frames(self):
        """Extract frames from video at specified intervals"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

        # Calculate interval to extract at most max_frames
        adaptive_interval = max(1, total_frames // self.max_frames)
        interval = max(self.frame_interval, adaptive_interval)

        print(f"Extracting frames with interval: {interval}")

        # Loop through video frames
        with tqdm(total=total_frames // interval) as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    self.frames.append(frame)
                    self.frame_indices.append(frame_idx)
                    pbar.update(1)

                frame_idx += 1

                # Limit the number of frames
                if len(self.frames) >= self.max_frames:
                    break

        cap.release()
        print(f"Extracted {len(self.frames)} frames")
        return self.frames, self.frame_indices

    def stitch_frames(self):
        """Stitch frames into a panorama and track frame positions"""
        if not self.frames:
            raise ValueError("No frames to stitch. Call extract_frames() first.")

        # Create OpenCV stitcher object
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS if self.mode == "scans" else cv2.Stitcher_PANORAMA)

        # Display progress bar for stitching
        print("Stitching frames...")
        # Since we can't track progress inside OpenCV's stitcher, we'll create a spinner
        # Use tqdm with a fake total that updates every 0.1 seconds
        with tqdm(total=100, desc="Initial stitching attempt",
                  bar_format='{desc}: {bar}| {percentage:3.0f}% [{elapsed}<{remaining}]') as pbar:
            # Start with 0%
            pbar.update(1)

            # Call the stitcher
            status, panorama = stitcher.stitch(self.frames)

            # Completed (either success or failure)
            pbar.update(99)

        if status != cv2.Stitcher_OK:
            # Try reducing the number of frames
            print(f"Initial stitching failed (status: {status}), trying with fewer frames...")

            # Calculate number of fallback attempts we might try
            remaining_attempts = sum(1 for s in [2, 4, 8] if len(self.frames[::s]) >= 3)

            # Create progress bar for fallback attempts
            with tqdm(total=remaining_attempts, desc="Fallback attempts", unit="try") as pbar:
                step = 2
                while status != cv2.Stitcher_OK and step <= 8:
                    reduced_frames = self.frames[::step]
                    if len(reduced_frames) < 3:  # Need at least 3 frames to stitch
                        break

                    print(f"Trying with {len(reduced_frames)} frames...")

                    # Progress bar for this specific attempt
                    with tqdm(total=100, desc=f"Attempt with 1/{step} frames",
                              bar_format='{desc}: {bar}| {percentage:3.0f}% [{elapsed}<{remaining}]',
                              leave=False) as sub_pbar:
                        sub_pbar.update(1)
                        status, panorama = stitcher.stitch(reduced_frames)
                        sub_pbar.update(99)

                    pbar.update(1)
                    step *= 2

        if status != cv2.Stitcher_OK:
            error_messages = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images for stitching",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
            }
            error_msg = error_messages.get(status, f"Unknown error (code: {status})")
            raise RuntimeError(f"Stitching failed: {error_msg}")

        print("Stitching successful!")

        # Extract homographies from stitcher (if available)
        if hasattr(stitcher, 'cameras'):
            cameras = stitcher.cameras
            for i, camera in enumerate(cameras):
                self.homographies.append(camera.R.copy())  # Store rotation matrix (simplified)

        # Store the stitched map
        self.stitched_map = panorama

        # Compute the locations of frame corners in the stitched image
        self.compute_frame_corners()

        return panorama

    def compute_frame_corners(self):
        """Compute the corners of each frame in the stitched map"""
        if self.stitched_map is None:
            raise ValueError("Stitched map not available. Call stitch_frames() first.")

        # Use feature matching to find frame locations in the stitched map
        print("Computing frame locations in the stitched map...")

        # Initialize SIFT detector for feature extraction
        sift = cv2.SIFT_create()

        # For each original frame
        for i, frame in enumerate(self.frames):
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_map = cv2.cvtColor(self.stitched_map, cv2.COLOR_BGR2GRAY)

            # Detect features
            kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
            kp_map, des_map = sift.detectAndCompute(gray_map, None)

            if des_frame is None or des_map is None:
                print(f"Warning: No features detected for frame {i}")
                self.frame_corners.append(None)
                continue

            # Match features
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des_frame, des_map, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                print(f"Warning: Not enough matches for frame {i}")
                self.frame_corners.append(None)
                continue

            # Get matched keypoints
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                print(f"Warning: Homography estimation failed for frame {i}")
                self.frame_corners.append(None)
                continue

            # Get frame corners
            h, w = frame.shape[:2]
            corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # Transform corners to stitched map coordinates
            transformed_corners = cv2.perspectiveTransform(corners, H)

            # Store corners
            self.frame_corners.append(transformed_corners.reshape(4, 2))

        return self.frame_corners

    def enhance_panorama(self):
        """Apply some enhancements to the stitched panorama"""
        if self.stitched_map is None:
            raise ValueError("Stitched map not available. Call stitch_frames() first.")

        # Convert to grayscale to find the borders
        gray = cv2.cvtColor(self.stitched_map, cv2.COLOR_BGR2GRAY)

        # Threshold to find black regions
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (the non-black area)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop to remove black borders
            self.stitched_map = self.stitched_map[y:y + h, x:x + w]

            # Adjust frame corners
            if self.frame_corners:
                for i in range(len(self.frame_corners)):
                    if self.frame_corners[i] is not None:
                        # Subtract origin of crop
                        self.frame_corners[i] = self.frame_corners[i] - [x, y]

        return self.stitched_map

    def save_results(self, output_image="stitched_map.jpg", output_data="frame_data.json"):
        """Save stitched map and frame location data"""
        if self.stitched_map is None:
            raise ValueError("Stitched map not available. Call stitch_frames() first.")

        # Save stitched map
        cv2.imwrite(output_image, self.stitched_map)
        print(f"Stitched map saved to {output_image}")

        # Save frame data
        data = {
            "video_path": self.video_path,
            "frame_count": len(self.frames),
            "frame_indices": self.frame_indices,
            "frame_corners": [corners.tolist() if corners is not None else None for corners in self.frame_corners]
        }

        with open(output_data, 'w') as f:
            json.dump(data, f)

        print(f"Frame data saved to {output_data}")

    def visualize_frame_locations(self, output_image="frame_locations.jpg"):
        """Visualize the locations of frames in the stitched map"""
        if self.stitched_map is None or not self.frame_corners:
            raise ValueError("Stitched map or frame corners not available.")

        # Create a copy of the stitched map for visualization
        vis_map = self.stitched_map.copy()

        # Draw frame outlines
        for i, corners in enumerate(self.frame_corners):
            if corners is not None:
                # Convert corners to integer coordinates
                corners = corners.astype(np.int32)

                # Draw polygon
                cv2.polylines(vis_map, [corners], True, (0, 255, 0), 2)

                # Add frame number
                centroid = np.mean(corners, axis=0).astype(int)
                cv2.putText(vis_map, f"{i}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save visualization
        cv2.imwrite(output_image, vis_map)
        print(f"Frame location visualization saved to {output_image}")

        return vis_map

    def project_annotation_to_frame(self, annotation, frame_idx):
        """
        Project annotation from stitched map to original frame

        Args:
            annotation: Polygon or circle annotation in stitched map coordinates
            frame_idx: Index of the frame to project to

        Returns:
            Annotation in frame coordinates
        """
        if frame_idx >= len(self.frames):
            raise ValueError(f"Frame index {frame_idx} out of range")

        if self.frame_corners[frame_idx] is None:
            raise ValueError(f"No mapping available for frame {frame_idx}")

        # This is a simplified implementation - would need to be expanded
        # based on your specific annotation format

        # For demonstration, assuming annotation is a polygon (list of points)
        # We would need to compute the inverse transformation from map to frame

        # Get the corners in the map
        map_corners = self.frame_corners[frame_idx]

        # Original frame corners
        h, w = self.frames[frame_idx].shape[:2]
        frame_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])

        # Compute perspective transform from map to frame
        H_inv = cv2.getPerspectiveTransform(
            map_corners.astype(np.float32),
            frame_corners.astype(np.float32)
        )

        # Transform annotation points
        annotation_points = np.array(annotation).reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(annotation_points, H_inv)

        return transformed_points.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description="Stitch video frames into a panorama map with frame tracking")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", "-o", default="stitched_map.jpg", help="Output image path")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Extract one frame every N frames")
    parser.add_argument("--max_frames", "-m", type=int, default=100, help="Maximum number of frames to extract")
    parser.add_argument("--mode", choices=["panorama", "scans"], default="scans",
                        help="Stitching mode: 'panorama' for 360° panorama, 'scans' for orthographic map")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize frames and result")

    args = parser.parse_args()

    try:
        # Create stitcher
        stitcher = VideoMapStitcher(
            args.video_path,
            frame_interval=args.interval,
            max_frames=args.max_frames,
            mode=args.mode
        )

        # Extract frames
        stitcher.extract_frames()

        if len(stitcher.frames) < 3:
            print("Error: Need at least 3 frames for stitching.")
            return

        # Stitch frames
        stitcher.stitch_frames()

        # Enhance panorama
        stitcher.enhance_panorama()

        # Save results
        base_name = os.path.splitext(args.output)[0]
        stitcher.save_results(
            output_image=args.output,
            output_data=f"{base_name}_data.json"
        )

        # Visualize frame locations
        vis_map = stitcher.visualize_frame_locations(output_image=f"{base_name}_frames.jpg")

        # Show results if requested
        if args.visualize:
            # Resize for visualization if too large
            height, width = vis_map.shape[:2]
            if width > 1200 or height > 800:
                scale = min(1200 / width, 800 / height)
                display_img = cv2.resize(vis_map, (0, 0), fx=scale, fy=scale)
            else:
                display_img = vis_map.copy()

            cv2.imshow("Frame Locations in Stitched Map", display_img)
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()