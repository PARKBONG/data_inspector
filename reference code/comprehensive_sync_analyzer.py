
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import wave
import argparse
from datetime import datetime

def parse_time(time_str):
    """Converts HH:MM:SS.mmm to float seconds."""
    try:
        h, m, s_ms = time_str.split(':')
        s, ms = s_ms.split('.')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    except Exception as e:
        # Fallback for slightly different formats if any
        return 0.0

def load_jsonl(path, time_key, val_keys):
    times = []
    data_dict = {k: [] for k in val_keys}
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return np.array([]), {k: np.array([]) for k in val_keys}
        
    with open(path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = parse_time(obj[time_key])
                times.append(t)
                for k in val_keys:
                    # Handle nested keys like "Pose.X"
                    parts = k.split('.')
                    val = obj
                    for p in parts:
                        val = val[p]
                    data_dict[k].append(val)
            except:
                continue
    return np.array(times), {k: np.array(v) for k, v in data_dict.items()}

def load_audio_intensity(wav_path, window_size_sec=0.02):
    if not os.path.exists(wav_path):
        print(f"Warning: Audio file not found {wav_path}")
        return np.array([]), np.array([])
        
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        
        if samp_width == 2:
            dtype = np.int16
        elif samp_width == 3:
            raw_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
            empty = np.zeros((raw_array.shape[0], 1), dtype=np.uint8)
            padded = np.hstack((raw_array, empty))
            data32 = padded.view(np.int32).flatten()
            data32 = np.where(data32 >= 2**23, data32 - 2**24, data32)
            audio_samples = data32.astype(np.float32)
        elif samp_width == 4:
            audio_samples = np.frombuffer(raw_data, dtype=np.float32)
        else:
            return np.array([]), np.array([])
            
        if samp_width != 3 and samp_width != 4:
            audio_samples = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

        if n_channels > 1:
            audio_samples = audio_samples[::n_channels]
            
        window_size = int(window_size_sec * framerate)
        n_windows = len(audio_samples) // window_size
        rms = []
        audio_times = []
        for i in range(n_windows):
            chunk = audio_samples[i*window_size : (i+1)*window_size]
            rms_val = np.sqrt(np.mean(chunk**2))
            rms.append(rms_val)
            audio_times.append((i + 0.5) * window_size_sec)
            
    return np.array(audio_times), np.array(rms)

def normalize(arr):
    if len(arr) == 0: return arr
    amin, amax = np.min(arr), np.max(arr)
    if amin == amax: return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)

def main(db_path):
    # Paths
    robot_file = os.path.join(db_path, "Robot", "0.jsonl")
    audio_file = os.path.join(db_path, "Audio", "0.wav")
    model_file = os.path.join(db_path, "Model", "0.jsonl")
    ir_file = os.path.join(db_path, "IR_High", "0.jsonl")
    
    print(f"Analyzing alignment for session: {os.path.basename(db_path)}")
    
    # Load Data
    t_robot, d_robot = load_jsonl(robot_file, "Time", ["ArcOn"])
    t_audio, d_audio = load_audio_intensity(audio_file)
    t_model, d_model = load_jsonl(model_file, "Time", ["LaserOn"])
    t_ir, d_ir       = load_jsonl(ir_file, "Time", ["MaxTemp"])
    
    # Plotting
    plt.figure(figsize=(16, 10))
    
    # 1. Robot ArcOn (Red)
    if len(t_robot) > 0:
        plt.step(t_robot, d_robot["ArcOn"], label="Robot: ArcOn", color='r', linewidth=2, alpha=0.8, where='post')
    
    # 2. Model LaserOn (Orange - dashed)
    if len(t_model) > 0:
        plt.step(t_model, d_model["LaserOn"], label="Model: LaserOn", color='orange', linestyle='--', linewidth=1.5, alpha=0.8, where='post')
        
    # 3. Audio RMS (Blue - Normalized)
    if len(t_audio) > 0:
        plt.plot(t_audio, normalize(d_audio), label="Audio: Intensity (Norm)", color='b', alpha=0.5)
        
    plt.ylabel("Normalized Value / Logic State", fontsize=12)
    plt.legend(loc='upper left', frameon=True)

    # 4. IR MaxTemp (Green - Line with Dots, Raw Value on Secondary Axis)
    if len(t_ir) > 0:
        ax2 = plt.gca().twinx()
        # Connect with lines and small markers
        ax2.plot(t_ir, d_ir["MaxTemp"], label="IR High: MaxTemp (Raw)", color='g', marker='o', markersize=1, linewidth=0.5, alpha=0.6)
        ax2.set_ylabel("Temperature (°C)", fontsize=12, color='g')
        ax2.set_ylim(0, 2400) # Set min 600, max 2000 (assuming 200 was typo for 2000)
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='upper right', frameon=True)

    # Intersection Detection (Events)
    if len(t_robot) > 0:
        robot_starts = t_robot[1:][np.diff(d_robot["ArcOn"].astype(int)) == 1]
        # for start in robot_starts:/
        #     plt.axvline(x=start, color='r', linestyle=':', alpha=0.4)

    plt.title(f"Comprehensive Data Alignment: {os.path.basename(db_path)}", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    
    save_path = os.path.join(db_path, "comprehensive_sync_analysis.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved analysis plot to: {save_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(current_dir, "DB", "260226_173131")

    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs='?', default=default_path, help="Path to DB session folder")
    args = parser.parse_args()
    
    if os.path.exists(args.db_path):
        main(args.db_path)
    else:
        print(f"Error: Path not found - {args.db_path}")
