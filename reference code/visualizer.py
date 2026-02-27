# import numpy as np
# import matplotlib.pyplot as plt
# import os, struct

# path = "DB/260211_180443/IR_High/0.raw"

# with open(path, "rb") as f:
#     # File header: width(int32), height(int32)
#     header = f.read(8)
#     W, H = struct.unpack("<ii", header)   # little-endian int32
#     print("W,H =", W, H)

#     frame_payload_bytes = 8 + (H * W * 2)  # timestamp(int64) + pixels(uint16)
#     file_size = os.path.getsize(path)

#     # Remaining bytes after file header
#     rem = file_size - 8
#     n_frames = rem // frame_payload_bytes
#     print("file_size =", file_size, "n_frames =", n_frames, "rem_mod =", rem % frame_payload_bytes)

#     # Read first frame
#     ts_bytes = f.read(8)
#     (ticks,) = struct.unpack("<q", ts_bytes)  # int64
#     img = np.frombuffer(f.read(H * W * 2), dtype=np.uint16).reshape(H, W)

# print("first ticks =", ticks)

# plt.figure()
# plt.imshow(img, cmap="inferno")
# plt.colorbar(label="raw uint16")
# plt.title("Frame 0")
# plt.show()

# import numpy as np, struct, os
# from datetime import datetime, timedelta

# path = "DB/260211_180443/IR_High/0.raw"

# with open(path, "rb") as f:
#     W, H = struct.unpack("<ii", f.read(8))
#     frame_bytes = 8 + H*W*2
#     file_size = os.path.getsize(path)
#     n = (file_size - 8) // frame_bytes

#     def read_ticks(i):
#         f.seek(8 + i*frame_bytes)
#         return struct.unpack("<q", f.read(8))[0]

#     t0 = read_ticks(0)
#     t1 = read_ticks(n-1)

# # .NET ticks: 100 ns since 0001-01-01
# TICKS_PER_SEC = 10_000_000
# duration_sec = (t1 - t0) / TICKS_PER_SEC
# fps = (n-1) / duration_sec

# print("W,H:", W, H)
# print("frames:", n)
# print("duration_sec:", duration_sec)
# print("effective_fps:", fps)

# import numpy as np, struct, os
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# path = "DB/260211_180443/IR_High/0.raw"

# with open(path, "rb") as f:
#     W, H = struct.unpack("<ii", f.read(8))
#     frame_bytes = 8 + H*W*2
#     file_size = os.path.getsize(path)
#     n_frames = (file_size - 8) // frame_bytes

#     def read_frame(i):
#         f.seek(8 + i*frame_bytes)
#         ticks = struct.unpack("<q", f.read(8))[0]
#         img = np.frombuffer(f.read(H*W*2), dtype=np.uint16).reshape(H, W)
#         return ticks, img

#     _, img0 = read_frame(0)

# fig, ax = plt.subplots()
# im = ax.imshow(img0, cmap="inferno", interpolation="nearest")
# plt.colorbar(im, ax=ax)

# def update(i):
#     ticks, img = read_frame(i)
#     im.set_data(img)
#     ax.set_title(f"{i}/{n_frames-1}  ticks={ticks}")
#     return (im,)

# ani = FuncAnimation(fig, update, frames=n_frames, interval=1, blit=False)
# plt.show()
import numpy as np, struct, os

path = r"DB/260211_180443/IR_High/0.raw"

with open(path, "rb") as f:
    W, H = struct.unpack("<ii", f.read(8))
    frame_bytes = 8 + H*W*2
    file_size = os.path.getsize(path)
    n_frames = (file_size - 8) // frame_bytes

    def read_frame(i):
        f.seek(8 + i*frame_bytes)
        ticks = struct.unpack("<q", f.read(8))[0]
        img = np.frombuffer(f.read(H*W*2), dtype=np.uint16).reshape(H, W)
        return ticks, img

def is_alive(img, delta=50):
    # delta: raw 기준. (필요시 10~200 사이로 조절)
    return (int(img.max()) - int(img.min())) > delta

# 1) 앞에서부터 첫 alive 찾기
start = None
with open(path, "rb") as f:
    W, H = struct.unpack("<ii", f.read(8))
    frame_bytes = 8 + H*W*2
    file_size = os.path.getsize(path)
    n_frames = (file_size - 8) // frame_bytes

    for i in range(n_frames):
        f.seek(8 + i*frame_bytes)
        f.read(8)
        img = np.frombuffer(f.read(H*W*2), dtype=np.uint16).reshape(H, W)
        if is_alive(img, delta=50):
            start = i
            break

# 2) 뒤에서부터 마지막 alive 찾기
end = None
with open(path, "rb") as f:
    W, H = struct.unpack("<ii", f.read(8))
    frame_bytes = 8 + H*W*2
    file_size = os.path.getsize(path)
    n_frames = (file_size - 8) // frame_bytes

    for i in range(n_frames - 1, -1, -1):
        f.seek(8 + i*frame_bytes)
        f.read(8)
        img = np.frombuffer(f.read(H*W*2), dtype=np.uint16).reshape(H, W)
        if is_alive(img, delta=50):
            end = i
            break

print("trim range:", start, "to", end, "kept:", end - start + 1)


import numpy as np, struct, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

path = r"DB/260211_180443/IR_High/0.raw"

# 1) open file (NOT in a with-block)
f = open(path, "rb")

W, H = struct.unpack("<ii", f.read(8))
frame_bytes = 8 + H * W * 2
file_size = os.path.getsize(path)
n_frames = (file_size - 8) // frame_bytes

def read_frame(i: int):
    f.seek(8 + i * frame_bytes)
    ticks = struct.unpack("<q", f.read(8))[0]
    img = np.frombuffer(f.read(H * W * 2), dtype=np.uint16).reshape(H, W)
    return ticks, img

_, img0 = read_frame(0)

fig, ax = plt.subplots()
vmin = 11000
vmax = 22000
im = ax.imshow(img0, cmap="inferno", interpolation="nearest", vmin=vmin, vmax=vmax)
cb = plt.colorbar(im, ax=ax)

def update(i):
    ticks, img = read_frame(i)

    im.set_data(img)

    # 🔥 핵심: 매 프레임마다 스케일 갱신
    # vmin = img.min()
    # vmax = img.max()
    im.set_clim(vmin, vmax)

    ax.set_title(f"{i}/{n_frames-1}")
    return (im,)

trim_start, trim_end = start, end

# FuncAnimation frames를 range로 제한
ani = FuncAnimation(fig, update, frames=range(trim_start, trim_end + 1),
                    interval=0.01, blit=False)
plt.show()

# 2) close after window is closed
f.close()