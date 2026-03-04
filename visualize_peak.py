from pathlib import Path
import argparse
import re
from typing import Dict
import matplotlib.pyplot as plt
from loader.session import SessionData

SESSION_PATTERN = re.compile(r"^\d{6}_\d{6}.*$")

def discover_sessions(db_root: Path) -> Dict[str, Path]:
    sessions: Dict[str, Path] = {}
    if not db_root.exists():
        return {}

    for path in db_root.rglob("*"):
        if path.is_dir() and SESSION_PATTERN.fullmatch(path.name):
            sessions[path.name] = path
    return dict(sorted(sessions.items()))

def build_peak_analysis_fig(session: SessionData):
    series = session.ir_high.json_series
    times = series.times
    peak_x = series.values.get("PeakX", [])
    peak_y = series.values.get("PeakY", [])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    if len(times) and len(peak_x):
        ax1.plot(times, peak_x, label="Peak X", color="#17becf", linewidth=1)
    ax1.set_title(f"Thermal Peak Analysis: {session.path.name}")
    ax1.set_ylabel("Peak X")
    ax1.grid(True, linestyle="--", alpha=0.6)

    if len(times) and len(peak_y):
        ax2.plot(times, peak_y, label="Peak Y", color="#ff7f0e", linewidth=1)
    ax2.set_ylabel("Peak Y")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig

def main():
    # settings
    db_root = Path("DB")
    session_id = None  # None이면 첫 번째 세션을 자동으로 선택합니다. (예: "260227_155409")

    session_paths = discover_sessions(db_root)
    
    if not session_paths:
        print(f"No sessions found under {db_root}")
        return

    if session_id is None:
        session_id = next(iter(session_paths))
        
    if session_id not in session_paths:
        print(f"Session {session_id} not found.")
        return

    print(f"Loading session: {session_id}")
    session = SessionData(session_paths[session_id])
    
    build_peak_analysis_fig(session)
    plt.show()

if __name__ == "__main__":
    main()
