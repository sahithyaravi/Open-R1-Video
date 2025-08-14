
from PIL import Image

import torch
import numpy as np


def adaptive_frame_sampling(
    scores,
    max_frames=32,
    vr=None,
    temperature=0.1, pick_mode="linspace"):
    """
    Adaptive frame sampling based on surprise scores.
    Args:
        scores (List[float]): List of surprise scores for each interval.
        max_frames (int): Maximum number of frames to sample.
        vr: Video reader object.
        temperature (float): Softmax temperature (not used here but could be).
        pick_mode (str): "linspace" or "random"
    Returns:
        List[Image]: Sampled frames as PIL images.
        List[int]: Corresponding frame indices.
    """
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps
    n_intervals = len(scores)

    intervals = np.linspace(0.0, duration, n_intervals + 1)
    base_allocation = np.ones(n_intervals, dtype=int)
    remaining_frames = max_frames - n_intervals
   
    if len(set(scores)) == 1:
        print("All scores are equal, distributing remaining frames uniformly.")
        extra_allocation = np.full(n_intervals, remaining_frames // n_intervals, dtype=int)
        for i in range(remaining_frames % n_intervals):
            extra_allocation[i] += 1
    else:
        scores_t = torch.tensor(scores, dtype=torch.float32, device="cpu")
        scores_np = np.array([s.item() for s in scores_t], dtype=float)
        probs = scores_np / (np.sum(scores_np) + 1e-8)  # Avoid division by zero
        extra_allocation = np.floor(probs * remaining_frames).astype(int)
        while extra_allocation.sum() < remaining_frames:
            residual = probs - extra_allocation
            idx = residual.argmax()
            extra_allocation[idx] += 1

    allocation = base_allocation + extra_allocation
    # # Use allocation to pick timestamps from intervals
    time_stamps = []
    for idx, n in enumerate(allocation):
        if n > 0:
            start_time = intervals[idx]
            end_time = intervals[idx + 1]
            if pick_mode == "linspace":
                ts = np.linspace(start_time, end_time, n, endpoint=False)
            elif pick_mode == "random":
                ts = np.random.uniform(start_time, end_time, n)
            else:
                raise ValueError(f"Unknown pick_mode: {pick_mode}")
            time_stamps.extend(ts)

    time_stamps = np.clip(time_stamps, 0, duration)
    frame_indices = np.unique((np.array(time_stamps) * fps).astype(int)).tolist()
    frames_nd = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(f) for f in frames_nd], frame_indices
   


def caption_frames(model, processor, frames, scores):
    """
    Generate captions for a video using the weighted captioning method.
    
    Args:
        model: The Qwen2.5 VL model.
        processor: The processor for the model.
        video_frames: List of PIL images representing video frames.
        scores: List of scores corresponding to each frame.
        max_length: Maximum length of the generated caption.
        num_beams: Number of beams for beam search.
        device: Device to run the model on ('cuda' or 'cpu').
        
    Returns:
        List of generated captions.
    """
    cap_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Summarize the key events in the video in 100 words."},
                {"type": "video"},
            ],
        }
    ]
    cap_prompt = processor.apply_chat_template(cap_conv, add_generation_prompt=True)
    cap_inputs = processor(
        text=cap_prompt,
        videos=frames,
        return_tensors="pt"
    )
    cap_ids = model.generate(**cap_inputs.to(model.device), max_new_tokens=200)
    caption = processor.batch_decode(cap_ids, skip_special_tokens=True)[0].lower().split("assistant\n")[1].replace(":", "")
    return caption

def caption_by_weight(model, processor, frames, scores, vr):
    """
    Generate captions for a video using the weighted captioning method.
    
    Args:
        model: The Qwen2.5 VL model.
        processor: The processor for the model.
        video_frames: List of PIL images representing video frames.
        scores: List of scores corresponding to each frame.
        
    Returns:
        List of generated captions.
    """
    sampled_frames, frame_indices = adaptive_frame_sampling(scores=scores, vr=vr)
    caption = caption_frames(model, processor, sampled_frames, scores)
    return caption, str(frame_indices)

