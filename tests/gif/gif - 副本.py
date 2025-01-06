from moviepy.editor import VideoFileClip,vfx


def speed_up_video(input_path, output_path, speed_factor):
    # Load the video file
    video = VideoFileClip(input_path)

    # Apply the speed-up effect
    sped_up_video = video.fx(vfx.speedx, speed_factor)

    # Save the sped-up video
    sped_up_video.write_videofile(output_path, codec='libx264')


if __name__ == "__main__":
    input_video_path = "real_video.mp4"  # Replace with your input video file path
    output_video_path = "1_real.mp4"  # Replace with your desired output file path
    speed_factor = 3.5  # Set the speed factor (e.g., 2.0 for 2x speed)

    speed_up_video(input_video_path, output_video_path, speed_factor)
