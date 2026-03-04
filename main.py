import argparse
import re
from pathlib import Path
from typing import Dict

from server.app import run_server


SESSION_PATTERN = re.compile(r"^\d{6}_\d{6}.*$")


def discover_sessions(db_root: Path) -> Dict[str, Path]:
    sessions: Dict[str, Path] = {}
    if not db_root.exists():
        raise FileNotFoundError(f"DB root not found: {db_root}")

    for path in db_root.rglob("*"):
        if path.is_dir() and SESSION_PATTERN.fullmatch(path.name):
            sessions[path.name] = path
    return dict(sorted(sessions.items()))


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive sensor inspector")
    parser.add_argument(
        "--db-root",
        default="DB",
        help="Root directory that contains session folders (default: DB)",
    )
    parser.add_argument(
        "--session",
        help="Initial session ID to load (format YYMMDD_HHMMSS). Defaults to the first discovered session.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    db_root = Path(args.db_root)
    session_paths = discover_sessions(db_root)
    if not session_paths:
        raise RuntimeError(f"No sessions found under {db_root}")

    default_session = args.session or next(iter(session_paths))
    if default_session not in session_paths:
        raise ValueError(f"Session {default_session} not found under {db_root}")

    run_server(session_paths, default_session, host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
