
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

# Mocking and importing
sys.path.append(str(Path.cwd()))
from loader.session import SessionData
from visualizer import figures

def diagnostic_comparison():
    session_path = Path(r"c:\Users\user\Desktop\data_inspector\DB\260227\260227_155409")
    session = SessionData(session_path)
    
    # Pick a timestamp where we know there's data
    # From my previous search, 23.223 had a peak and contour
    ts = 23.223
    metadata = session.get_ir_high_metadata(ts)
    
    print(f"--- Diagnostic for timestamp {ts} ---")
    print(f"Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
    if metadata:
        print(f"ContourPoints length: {len(metadata.get('ContourPoints', []))}")
        print(f"Peak: ({metadata.get('PeakX')}, {metadata.get('PeakY')})")
        print(f"Area: {metadata.get('Area')}")

    # Build figures
    fig_main = figures.build_ir(session.ir_high.raw_reader, ts, "IR High (Main)", metadata=metadata, draw_contour=True, draw_peak=True)
    fig_test = figures.build_ir(session.ir_high.raw_reader, ts, "Test: Contour Only", metadata=metadata, draw_contour=True, draw_peak=False)
    
    def inspect_fig(name, fig):
        print(f"\n[{name}]")
        print(f"Number of traces: {len(fig.data)}")
        for i, trace in enumerate(fig.data):
            trace_type = trace.type
            trace_name = getattr(trace, 'name', 'N/A')
            print(f"  Trace {i}: type={trace_type}, name='{trace_name}'")
            if isinstance(trace, go.Scatter):
                print(f"    Scatter points: {len(trace.x) if trace.x is not None else 0}")
                if len(trace.x) > 0:
                    print(f"    First X,Y: ({trace.x[0]}, {trace.y[0]})")
            elif isinstance(trace, go.Heatmap):
                print(f"    Heatmap Z shape: {np.array(trace.z).shape}")
    
    inspect_fig("Main", fig_main)
    inspect_fig("Test", fig_test)
    
    # Check for layout differences
    print("\n[Layout Check]")
    print(f"Main title: {fig_main.layout.title.text}")
    print(f"Main Y-axis autorange: {fig_main.layout.yaxis.autorange}")

if __name__ == "__main__":
    diagnostic_comparison()
