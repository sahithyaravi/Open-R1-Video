from decord import VideoReader, cpu
import numpy as np



def extract_k_frames_decord_cpu(video_path: str, k: int = 30):
    vr = VideoReader(video_path, ctx=cpu(0))  # explicitly use CPU
    total_frames = len(vr)
    print(f"Total frames in video: {total_frames}")
    indices = np.linspace(0, total_frames - 1, k, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # shape: (k, H, W, 3)
    fps           = vr.get_avg_fps()
    return [frame for frame in frames], indices, total_frames, fps, vr