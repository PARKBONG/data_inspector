from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import base64

from .utils import timestamp_to_seconds


@dataclass
class ImageFrame:
    timestamp: float
    path: Path


class RGBLoader:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.frames: List[ImageFrame] = []
        if self.folder.exists():
            self._index_frames()

    def _index_frames(self):
        for image_path in sorted(self.folder.glob("*.jpg")):
            parts = image_path.stem.split("_")
            if len(parts) != 4:
                continue
            time_token = f"{parts[0]}:{parts[1]}:{parts[2]}.{parts[3]}"
            try:
                timestamp = timestamp_to_seconds(time_token)
            except ValueError:
                continue
            self.frames.append(ImageFrame(timestamp=timestamp, path=image_path))

    def find_nearest(self, timestamp: float) -> Optional[ImageFrame]:
        if not self.frames:
            return None
        return min(self.frames, key=lambda frame: abs(frame.timestamp - timestamp))

    @staticmethod
    def encode_image(path: Path) -> str:
        data = path.read_bytes()
        return base64.b64encode(data).decode("ascii")
