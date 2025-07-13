import cv2
import os
import argparse
from tqdm import tqdm


def video_to_images(video_path, output_dir, frame_step=1):
    # Check if video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, {fps:.2f} fps, {frame_count} frames")
    print(f"Saving every {frame_step} frame(s)")

    # Extract frames
    frame_idx = 0
    saved_count = 0
    # Create a progress bar
    with tqdm(total=frame_count, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only save frames according to frame_step
            if frame_idx % frame_step == 0:
                # Create formatted filename with 4-digit padding (e.g., 0001.jpg)
                filename = f"{saved_count:04d}.jpg"
                output_path = os.path.join(output_dir, filename)

                # Save the frame as an image
                cv2.imwrite(output_path, frame)
                saved_count += 1

            frame_idx += 1
            # Update progress bar
            pbar.update(1)

    # Release video resource
    cap.release()
    print(f"Completed: Extracted {saved_count} frames from {frame_idx} total frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert video to sequence of images')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('--frame_step', type=int, default=1, 
                        help='Save every Nth frame (default: 1, which saves every frame)')
    args = parser.parse_args()

    try:
        video_to_images(args.video_path, args.output_dir, args.frame_step)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
