
import sys
from pathlib import Path
import numpy as np

# Mocking the loader base or importing directly if possible
# Assuming current directory is project root
sys.path.append(str(Path.cwd()))

from loader.ir import IRRawLoader

def verify_ir_loader():
    raw_path = Path(r"c:\Users\user\Desktop\data_inspector\DB\260227\260227_155409\IR_High\0.raw")
    temp_range = (1100, 2000)
    
    print(f"Checking raw file: {raw_path}")
    if not raw_path.exists():
        print("ERROR: Raw file does not exist!")
        return
    
    loader = IRRawLoader(raw_path, temp_range)
    print(f"Frame Count: {loader.frame_count}")
    print(f"Width: {loader.width}, Height: {loader.height}")
    print(f"Timestamps length: {len(loader.timestamps)}")
    if len(loader.timestamps) > 0:
        print(f"First timestamp: {loader.timestamps[0]}")
        print(f"Last timestamp: {loader.timestamps[-1]}")
    
    # Check if a specific timestamp can fetch a frame
    # The middle timestamp
    mid_ts = loader.timestamps[len(loader.timestamps)//2] if len(loader.timestamps) > 0 else 0
    frame = loader.get_nearest_frame(mid_ts)
    if frame:
        print(f"Successfully fetched frame at {mid_ts}")
        print(f"Frame data shape: {frame.data.shape}")
        print(f"Frame min: {frame.data.min()}, max: {frame.data.max()}")
    else:
        print("ERROR: Could not fetch frame.")

if __name__ == "__main__":
    verify_ir_loader()
