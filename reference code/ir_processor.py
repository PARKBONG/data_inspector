import os
import struct
import json
import numpy as np
import cv2
from datetime import datetime, timedelta


class IR_Processor:
    def __init__(self, raw_path, output_dir, temp_min=1100, temp_max=2000):
        """
        Initialize the IR Processor.
        
        :param raw_path: Path to the .raw file.
        :param output_dir: Directory where processed PNGs will be saved.
        :param temp_min: Minimum temperature for colormap normalization (default 1100).
        :param temp_max: Maximum temperature for colormap normalization (default 2000).
        """
        self.raw_path = raw_path
        self.output_dir = output_dir
        self.temp_range = (temp_min, temp_max)
        
        # Internal metadata (extracted from header)
        self.width = 0
        self.height = 0
        self.total_size = os.path.getsize(raw_path) if os.path.exists(raw_path) else 0

    @staticmethod
    def ms_to_timestamp(ms):
        """Converts milliseconds to HH_mm_ss_fff string format for filenames."""
        td = timedelta(milliseconds=ms)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = ms % 1000
        return f"{hours:02d}_{minutes:02d}_{seconds:02d}_{milliseconds:03d}"

    def verify_config(self):
        """Reads the file header and returns basic file info."""
        if not os.path.exists(self.raw_path):
            print(f"Error: File not found - {self.raw_path}")
            return None

        with open(self.raw_path, 'rb') as f:
            header = f.read(8)
            if len(header) < 8:
                print("Error: Incomplete header")
                return None
            
            self.width, self.height = struct.unpack('<II', header)
            
        print(f"--- IR Raw File Header ---")
        print(f"Path: {self.raw_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Size: {self.total_size / (1024*1024):.2f} MB")
        print(f"Normalization Range: {self.temp_range[0]} - {self.temp_range[1]} C")
        print(f"Timestamp format: 4-byte Int (ms)")
        print(f"--------------------------")
        
        return {"width": self.width, "height": self.height}

    def process_to_png(self, skip_unvarying=True, colormap=cv2.COLORMAP_MAGMA):
        """
        Extracts frames from the raw file and saves them as colormapped PNGs.
        """
        if self.width == 0 or self.height == 0:
            if not self.verify_config():
                return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print(f"Extracting frames to: {self.output_dir}")
        
        frame_count = 0
        saved_count = 0
        
        with open(self.raw_path, 'rb') as f:
            f.seek(8)  # Skip header (Width, Height)
            
            while True:
                # 1. Read Millisecond Timestamp (4 bytes integer)
                ts_data = f.read(4)
                if not ts_data or len(ts_data) < 4:
                    break
                
                ms = struct.unpack('<i', ts_data)[0]
                timestamp_str = self.ms_to_timestamp(ms)
                
                # 2. Read Pixel Data (uint16)
                pixel_count = self.width * self.height
                raw_pixels = f.read(pixel_count * 2)
                if len(raw_pixels) < pixel_count * 2:
                    print(f"Incomplete data at {timestamp_str}")
                    break
                
                # 3. Process Image
                image = np.frombuffer(raw_pixels, dtype=np.uint16).reshape((self.height, self.width))
                
                # Filter: skip frames where all pixels are the same
                if skip_unvarying and np.all(image == image[0, 0]):
                    frame_count += 1
                    continue
                
                # 4. Normalize and Apply Colormap
                vmin, vmax = self.temp_range
                norm_img = np.clip(image, vmin, vmax)
                norm_img = ((norm_img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                color_img = cv2.applyColorMap(norm_img, colormap)
                
                # 5. Save PNG
                output_path = os.path.join(self.output_dir, f"{timestamp_str}.png")
                cv2.imwrite(output_path, color_img)
                
                frame_count += 1
                saved_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} / Saved {saved_count} frames...")

        print(f"Done! Processed {frame_count} frames, saved {saved_count} valid frames.")

    def process_jsonl_to_contour_png(self, jsonl_path, contour_dir):
        """
        Reads a JSONL file and generates contour images on a black background
        with a yellow dot at the peak location.
        """
        if self.width == 0 or self.height == 0:
            if not self.verify_config():
                # If raw file is missing, we need resolution from somewhere else.
                # Default for this sensor is 72x56.
                self.width, self.height = 72, 56

        if not os.path.exists(contour_dir):
            os.makedirs(contour_dir)

        print(f"Generating contour images from {jsonl_path} to {contour_dir}")
        
        saved_count = 0
        
        if not os.path.exists(jsonl_path):
            print(f"Error: JSONL not found - {jsonl_path}")
            return

        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    time_str = data.get("Time", "")
                    if not time_str:
                        continue
                        
                    # Convert "00:00:15.363" to "00_00_15_363"
                    timestamp_filename = time_str.replace(":", "_").replace(".", "_")
                    
                    max_temp = data.get("MaxTemp", 0)
                    contour_pts = data.get("ContourPoints", [])
                    peak_x = data.get("PeakX", 0)
                    peak_y = data.get("PeakY", 0)

                    # Only save if there are contour points
                    if not contour_pts or len(contour_pts) == 0:
                        continue

                    # Create black background
                    img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                    # Draw contour (white)
                    pts = np.array(contour_pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

                    # Draw peak (yellow dot)
                    if max_temp > self.temp_range[0]: # Only draw if there's actual signal
                        cv2.circle(img, (int(peak_x), int(peak_y)), radius=0, color=(0, 255, 255), thickness=-1)

                    # Save image
                    output_path = os.path.join(contour_dir, f"{timestamp_filename}.png")
                    cv2.imwrite(output_path, img)
                    saved_count += 1

                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue

        print(f"Done! Saved {saved_count} contour frames.")

if __name__ == "__main__":
    # Example usage for verification (Change path if needed)
    session_id = "260226_173131"
    target = "high"

    temperature = {"high": {"min": 1100, "max": 2000}, "low": {"min": 100, "max": 950}}
    current_dir = os.path.dirname(os.path.abspath(__file__))

    processor = IR_Processor(
        raw_path=fr'{current_dir}\DB\{session_id}\IR_{target}\0.raw',
        output_dir=fr'{current_dir}\DB\{session_id}_processed\IR_{target}',
        temp_min=temperature[target]["min"],
        temp_max=temperature[target]["max"]
    )
    
    processor.process_to_png()

    # Process Contours from JSONL
    jsonl_path = os.path.join(current_dir, 'DB', session_id, f'IR_{target.capitalize()}', '0.jsonl')
    contour_dir = os.path.join(current_dir, 'DB', session_id + '_processed', f'IR_{target.capitalize()}_contour')
    
    processor.process_jsonl_to_contour_png(jsonl_path, contour_dir)
