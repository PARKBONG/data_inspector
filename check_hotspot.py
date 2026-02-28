
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path.cwd()))
from loader.session import SessionData

def check_hotspot():
    session_path = Path(r"c:\Users\user\Desktop\data_inspector\DB\260227\260227_155409")
    session = SessionData(session_path)
    
    ts = 23.223
    metadata = session.get_ir_high_metadata(ts)
    frame = session.ir_high.raw_reader.frame_values(ts)
    
    if frame is None or metadata is None:
        print("Data missing")
        return
        
    peak_x = metadata.get("PeakX")
    peak_y = metadata.get("PeakY")
    
    # Find theoretical hotspot in frame data
    # frame is (56, 72) i.e. (H, W)
    max_idx = np.unravel_index(np.argmax(frame), frame.shape)
    # max_idx is (row, col) i.e. (y, x)
    
    print(f"Timestamp: {ts}")
    print(f"Metadata Peak: x={peak_x}, y={peak_y}")
    print(f"Raw Max Value: {np.max(frame)}")
    print(f"Raw Max Index (y_raw, x_raw): {max_idx}")
    
    # Check if (y_raw, x_raw) matches (peak_y, peak_x)
    if int(peak_y) == max_idx[0] and int(peak_x) == max_idx[1]:
        print("MATCH! (Top-down alignment confirmed, no flip needed)")
    elif int(55 - peak_y) == max_idx[0] and int(peak_x) == max_idx[1]:
        print("MATCH! (Bottom-up alignment confirmed, flip IS needed)")
    else:
        print("NO DIRECT MATCH. Possible off-by-one or metadata/raw discrepancy.")

if __name__ == "__main__":
    check_hotspot()
