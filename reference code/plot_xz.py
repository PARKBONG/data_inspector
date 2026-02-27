import json
import matplotlib.pyplot as plt
import os

def plot_xz_graph(jsonl_path, output_image_path):
    x_coords = []
    z_coords = []
    
    if not os.path.exists(jsonl_path):
        print(f"Error: File not found - {jsonl_path}")
        return

    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'X' in data and 'Z' in data:
                        x_coords.append(data['X'])
                        z_coords.append(data['Z'])
        
        if not x_coords:
            print("No data points found to plot.")
            return

        plt.figure(figsize=(12, 8))
        plt.plot(x_coords, z_coords, color='blue', linewidth=1.5, label='XZ Path')
        
        # Set same scale for both axes
        plt.axis('equal')
        
        plt.xlabel('X Coordinate (mm)')
        plt.ylabel('Z Coordinate (mm)')
        plt.title('XZ Graph (Deposition Path) - Equal Scale')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.savefig(output_image_path)
        print(f"Graph saved successfully to: {output_image_path}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    session_id = "260226_173131"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_file = os.path.join(current_dir, 'DB', session_id, 'Model', '0.jsonl')
    output_img = os.path.join(current_dir, 'DB', session_id + '_XZ_Graph_Equal.png')
    
    plot_xz_graph(jsonl_file, output_img)
