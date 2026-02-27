
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import wave
import argparse
from datetime import datetime

def parse_time(time_str):
    # Format: HH:MM:SS.mmm
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def load_robot_data(jsonl_path):
    times = []
    arc_ons = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                times.append(parse_time(data['Time']))
                arc_ons.append(1 if data['ArcOn'] else 0)
            except:
                continue
    return np.array(times), np.array(arc_ons)

def load_audio_intensity(wav_path, window_size_sec=0.05):
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # Read frames
        raw_data = wf.readframes(n_frames)
        
        # Determine dtype
        if samp_width == 2:
            dtype = np.int16
        elif samp_width == 3:
            # Handle 24-bit by padding
            raw_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
            # Pad to 32 bit (little endian)
            empty = np.zeros((raw_array.shape[0], 1), dtype=np.uint8)
            padded = np.hstack((raw_array, empty))
            # Interpret as int32 and shift (24-bit is usually signed)
            data32 = padded.view(np.int32).flatten()
            # Sign correction for 24-bit
            data32 = np.where(data32 >= 2**23, data32 - 2**24, data32)
            # Normalize or just keep scale
            audio_samples = data32.astype(np.float32)
        elif samp_width == 4:
            dtype = np.float32 # If it's float 32
        else:
            raise ValueError(f"Unsupported sample width: {samp_width}")
            
        if samp_width != 3:
            audio_samples = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

        # Extract mono if stereo
        if n_channels > 1:
            audio_samples = audio_samples[::n_channels]
            
        # Calculate RMS
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

def main(db_path):
    robot_path = os.path.join(db_path, "Robot", "0.jsonl")
    audio_path = os.path.join(db_path, "Audio", "0.wav")
    
    print(f"Loading files from {db_path}...")
    
    robot_times, arc_ons = load_robot_data(robot_path)
    audio_times, rms = load_audio_intensity(audio_path)
    
    # Normalize RMS for visualization
    if len(rms) > 0:
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    else:
        rms_norm = rms
        
    plt.figure(figsize=(15, 8))
    
    # Plot Robot ArcOn
    plt.step(robot_times, arc_ons, label="Robot ArcOn (0/1)", where='post', color='r', linewidth=2)
    
    # Plot Audio RMS
    plt.plot(audio_times, rms_norm, label="Audio Intensity (Normalized RMS)", alpha=0.7, color='b')
    
    # Identify transitions
    arc_start_idx = np.where(np.diff(arc_ons) == 1)[0]
    arc_end_idx = np.where(np.diff(arc_ons) == -1)[0]
    
    for idx in arc_start_idx:
        t = robot_times[idx+1]
        plt.axvline(x=t, color='g', linestyle='--', alpha=0.5)
        plt.text(t, 1.05, f"Start: {t:.2f}s", color='g', ha='center')
        
    for idx in arc_end_idx:
        t = robot_times[idx+1]
        plt.axvline(x=t, color='m', linestyle='--', alpha=0.5)
        plt.text(t, -0.1, f"End: {t:.2f}s", color='m', ha='center')

    plt.title(f"Audio-Robot Sync Analysis: {os.path.basename(db_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(db_path, "audio_robot_alignment.png")
    plt.savefig(save_path)
    print(f"Alignment plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", help="Path to the DB folder (e.g., DB/260226_173131)")
    args = parser.parse_args()
    
    main(args.db_path)
