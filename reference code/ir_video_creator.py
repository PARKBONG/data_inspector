import cv2
import os
import re
import numpy as np
from datetime import datetime

class IR_VideoCreator:
    def __init__(self, image_dir, output_video_path, fps=30):
        """
        Initialize the Video Creator.
        
        :param image_dir: Directory containing the extracted PNG frames.
        :param output_video_path: Path to save the resulting .mp4 or .avi.
        :param fps: Target frames per second for the video.
        """
        self.image_dir = image_dir
        self.output_video_path = output_video_path
        self.fps = fps

    def _get_ms_from_filename(self, filename):
        """Parses HH_mm_ss_fff.png to total milliseconds."""
        match = re.search(r'(\d+)_(\d+)_(\d+)_(\d+)', filename)
        if match:
            h, m, s, ms = map(int, match.groups())
            return h * 3600000 + m * 60000 + s * 1000 + ms
        return None

    def create_video(self):
        # 1. Get and sort images by timestamp
        images = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        if not images:
            print(f"No images found in {self.image_dir}")
            return

        # Sort based on internal milliseconds to ensure correct sequence
        images_with_time = []
        for img in images:
            ms = self._get_ms_from_filename(img)
            if ms is not None:
                images_with_time.append((ms, img))
        
        images_with_time.sort() # Sort by ms
        
        if not images_with_time:
            print("Could not parse timestamps from filenames.")
            return

        first_ms = images_with_time[0][0]
        last_ms = images_with_time[-1][0]
        total_duration_ms = last_ms - first_ms
        total_frames = int((total_duration_ms / 1000.0) * self.fps) + 1

        print(f"Creating video: {self.output_video_path}")
        print(f"Duration: {total_duration_ms/1000:.2f}s, Target FPS: {self.fps}")

        # 2. Initialize VideoWriter
        # Read the first image to get dimensions
        first_img_path = os.path.join(self.image_dir, images_with_time[0][1])
        sample_img = cv2.imread(first_img_path)
        height, width, layers = sample_img.shape
        
        # Scaling up for better visibility of text if resolution is very small (like 72x56)
        scale_factor = 10 if width < 100 else 1
        new_width, new_height = width * scale_factor, height * scale_factor

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
        video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (new_width, new_height))

        # 3. Generate frames based on target time to ensure accurate timing
        img_idx = 0
        for i in range(total_frames):
            target_ms = first_ms + (i * 1000.0 / self.fps)
            
            # Find the image closest to the target_ms (not exceeding it usually, or just nearest)
            while img_idx < len(images_with_time) - 1 and images_with_time[img_idx+1][0] <= target_ms:
                img_idx += 1
            
            # Current image info
            current_ms, current_filename = images_with_time[img_idx]
            timestamp_label = current_filename.replace('.png', '').replace('_', ':')
            
            # Read and process image
            frame = cv2.imread(os.path.join(self.image_dir, current_filename))
            
            if scale_factor > 1:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            # Overlay Text (Yellow)
            # Position: Top-left
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 * (new_width / 400.0) if new_width > 400 else 0.5
            thickness = 1 if new_width < 400 else 2
            color = (0, 255, 255) # Yellow in BGR
            
            cv2.putText(frame, timestamp_label, (10, int(new_height * 0.1)), font, font_scale, color, thickness, cv2.LINE_AA)
            
            video.write(frame)
            
            if i % 100 == 0:
                print(f"Writing frame {i}/{total_frames}...")

        video.release()
        print(f"Video saved successfully: {self.output_video_path}")

if __name__ == "__main__":
    session_id = "260226_173131"
    target = "High" # Consistent casing with directory names
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = fr'{current_dir}\DB\{session_id}_processed\IR_{target}'
    out_video = fr'{current_dir}\DB\{session_id}_processed\IR_{target}_video.mp4'
    
    # Increase FPS for smoother playback if the original source was high frequency
    creator = IR_VideoCreator(img_dir, out_video, fps=300)
    creator.create_video()

    # Create Video for Contour
    target = "High"
    img_dir = fr'{current_dir}\DB\{session_id}_processed\IR_{target}_contour'
    out_video = fr'{current_dir}\DB\{session_id}_processed\IR_{target}_Contour_video.mp4'
    
    if os.path.exists(img_dir):
        creator = IR_VideoCreator(img_dir, out_video, fps=300) # Using 20 FPS for clear view
        creator.create_video()
    else:
        print(f"Error: Directory not found - {img_dir}")
