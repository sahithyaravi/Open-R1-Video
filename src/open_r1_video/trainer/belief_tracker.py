import torch

from torch.nn.functional import softmax
from typing import List, Dict
from PIL import Image

from tqdm import tqdm

import os
import math

import gc
import numpy as np

from .weighted_captioning import caption_by_weight



def summarize_memory(memory_text: str) -> str:
    """
    Summarizes the memory text to fit within the model's context window.
    Args:
        memory_text: The text to summarize.
    Returns:
        Summarized text.
    """
    if len(memory_text.split()) <= 200:
        return memory_text
    else:
        # return max 200 words from the memory text
        return ' '.join(memory_text.split()[:200])
    
def generate_hypothesis(conv, visual_context, num_generations=3, model=None, processor=None):
    prompt = processor.apply_chat_template(
    conv, add_generation_prompt=True, )
    inputs = processor(text=prompt, videos=visual_context, return_tensors="pt",             
            padding=True,
            padding_side="left",
            add_special_tokens=False,)
    print(f"Generating {num_generations} hypotheses based on the prompt: {prompt}")
    print("Shape of each tensor in inputs:", {k: v.shape for k, v in inputs.items()})

    max_prompt_length = 8192

    inputs["input_ids"] = inputs["input_ids"][:, -max_prompt_length :]
    inputs["attention_mask"] = inputs["attention_mask"][:, -max_prompt_length :]

    outputs = model.generate(
        **inputs.to(model.device),
        do_sample=True,
        temperature=1.0,
        top_k=50,
        num_return_sequences=num_generations,
        max_new_tokens=40,
        pad_token_id=processor.pad_token_id,
    )
    hypotheses = processor.batch_decode(outputs, skip_special_tokens=True)
    # print(f"Generated {len(hypotheses)} hypotheses.")
    hypotheses = [hyp.strip().lower().split("assistant\n")[1].replace(":", "") for hyp in hypotheses]
    torch.cuda.empty_cache()
    gc.collect()
    return hypotheses


def caption_frame(context_frames: List[Image.Image], observed_frame: Image.Image, memory_text: str, model=None, processor=None) -> str:
    # Observe the actual frame and caption it. This is used to build working memory.
    cap_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"Here is what has happened so far: {memory_text}"},
                {"type": "text",
                 "text": "Describe what is happening in the last and most recent frame."},
                {"type": "video"},
                {"type": "image"},
            ],
        }
    ]
    cap_prompt = processor.apply_chat_template(cap_conv, add_generation_prompt=True)
    cap_inputs = processor(
        text=cap_prompt,
        videos=context_frames,
        images=[observed_frame],
        return_tensors="pt"
    )
    cap_ids = model.generate(**cap_inputs.to(model.device), max_new_tokens=30)
    caption = processor.batch_decode(cap_ids, skip_special_tokens=True)[0].lower().split("assistant\n")[1].replace(":", "")
    return caption


def qwen_bayesian_surprise_text_future(memory_text: str, context_frames: List[Image.Image], observed_frame: Image.Image, num_hypotheses: int, model=None, processor=None) -> Dict:
    """
    Computes Bayesian surprise using the Qwen model on visual data.
    Args:
        memory_text: Textual memory to provide context.
        context_frames: List of PIL images representing the context frames.
        observed_frame: The current frame to analyze.
        num_hypotheses: Number of hypotheses to generate.
    Returns:
        Dictionary containing surprise scores, hypotheses, priors, posteriors, and memory evolution.
    """

    # Sample hypotheses based on W and H
    conv = [
        {
            "role": "user", 
            "content": [{"type": "text", "text" : f"Here is what happened so far from the beginning of the video: {memory_text}"},
                        {"type": "text", "text": f"Based on this information, and recent frames from the video, answer in one sentence what will most likely happen in the next frame."},
                        {"type": "video"}],
        }
    ]
    # print(conv)
    # print(f"Generating {num_hypotheses} hypotheses based on memory and context frames.")
    # print(f"Memory text: {memory_text}")
    # print(f"Context frames: {len(context_frames)} frames")
    # print(f"Observed frame: type {type(observed_frame)}, size {observed_frame.size}")
    # print(f"Model: {model.__class__.__name__ if model else 'None'}")
    # print(f"Processor: {processor.__class__.__name__ if processor else 'None'}")
    h0 = generate_hypothesis(conv, context_frames, num_generations=num_hypotheses, model=model, processor=processor)
    print("Generated hypotheses based on memory and context frames:")
    for i, hyp in enumerate(h0):
        print(f"Hypothesis {i+1}: {hyp}")
    print("Hypotheses:", h0)

    # Sample hypotheses based on W, H and O
    conv = [
        {
            "role": "user",
            "content": [{"type": "text", "text" : f"Here is what happened so far from the beginning of the video: {memory_text}"},
                        {"type": "text", "text": "You are also provided with recent frames from the video. Answer in one sentence what is happening now."},
                        {"type": "video"},],
        }
    ]
    h1 = generate_hypothesis(conv, context_frames + [observed_frame], num_generations=num_hypotheses, model=model, processor=processor)
    hypotheses = h0 + h1

    # Compute P_Prior
    prior_scores = []
    prefix_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                "text": f"Here is what has happened so far: {memory_text}"},
                {"type": "video"},
                {"type": "text",
                "text": "Here is what will happen next:"},   # neutral anchor line
            ],
        }
    ]
    prefix_prompt = processor.apply_chat_template(
        prefix_conv, add_generation_prompt=False
    )
    prefix_inputs = processor(
        text=prefix_prompt,
        videos=context_frames,          # the k past frames
        return_tensors="pt"
    )
    prefix_len = prefix_inputs.input_ids.size(1)        # tokens in the input prompt
    # empty cuda cache
    torch.cuda.empty_cache()
    for hyp in hypotheses:
        full_txt   = prefix_prompt + hyp 
        full_input = processor(
            text=full_txt,
            videos=context_frames,      # same frames each time
            return_tensors="pt"
        )

        # mask: ignore input prompt or prefix tokens in the loss
        labels = full_input.input_ids.clone()
        labels[:, :prefix_len] = -100

        with torch.no_grad():
            loss = model(**full_input.to(model.device), labels=labels).loss
        prior_scores.append(-loss.item())                
        del full_input, labels, loss
        torch.cuda.empty_cache()
        gc.collect()
    log_prior_raw = torch.tensor(prior_scores, device=model.device)     # negative log likelihoods
    log_prior = log_prior_raw - torch.logsumexp(log_prior_raw, dim=0)   # log P(h) - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    P_prior   = log_prior.exp()      

    # Compute posterior
    # convert_tokens_to_ids when tokens may contain a leading space.
    def _single_id(tok: str) -> int:
        ids = processor.tokenizer.encode(tok, add_special_tokens=False)
        assert len(ids) == 1, f"‘{tok}’ splits into {ids}"
        return ids[0]

    yes_id = _single_id(" yes")
    no_id  = _single_id(" no")
    log_like = []
    all_frames = context_frames + [observed_frame]

    for hyp in hypotheses:
        conv_obs = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                    "text": f"Here is what has happened so far: {memory_text}"},
                    {"type": "video"},
                    {"type": "text",
                    "text": f"Statement: {hyp}\nIs this statement true in the CURRENT frame? "
                            "Answer 'yes' or 'no'."},
                ],
            }
        ]

        prompt_obs = processor.apply_chat_template(conv_obs, add_generation_prompt=True)
        # print(repr(prompt_obs[-120:]))      
        inp_obs = processor(
            text=prompt_obs,
            videos=all_frames,                
            return_tensors="pt"
        )

        with torch.no_grad():
            # logits shape: (batch_size, sequence_length, vocab_size)
            logits = model(**inp_obs.to(model.device)).logits[0, -1] 


        sub_logits = logits[[yes_id, no_id]]             
        logprob_yes = sub_logits.log_softmax(dim=-1)[0]    # log P(yes|hyp, obs)
        log_like.append(logprob_yes)
        del inp_obs, logits
        torch.cuda.empty_cache()
        gc.collect()


    log_like   = torch.stack(log_like)                                  # (H,)
    log_unnorm = log_prior + log_like                                   # log P*Z - Bayes rule
    log_post   = log_unnorm - torch.logsumexp(log_unnorm, dim=0)        # log P(h|obs) P_post = torch.softmax(log_unnorm, dim=0)
    P_post     = log_post.exp()


    kl  = torch.sum(P_post * (log_post - log_prior)).item()

    # kl_torch = kl_div(
    #     input = log_post,                 # log-probs
    #     target = log_prior.exp(),         # plain probs
    #     reduction = "sum",                # ∑_h  p_post * (log p_post − log p_prior)
    #     log_target = False                # because target is in probability space
    #  ).item()
    
    log_mix = torch.logaddexp(log_prior, log_post) - math.log(2.0)      # log M
    js  = 0.5 * (
            torch.sum(P_prior * (log_prior - log_mix)) +
            torch.sum(P_post  * (log_post  - log_mix))
        )
    js_norm = (js / math.log(2.0)).item()

    # --- per-belief (signed) JS contribution ------------------------------------
    d_js = 0.5 * (P_prior * (log_prior - log_mix) +
                P_post  * (log_post  - log_mix))           # tensor, same length as hypotheses


    
    # add prior, posterior as string to each hypothesis in hypotheses
    hypotheses = [
        f"{hyp} (prior: {P_prior[i]:.3f}, posterior: {P_post[i]:.3f}), JS: {d_js[i]:.3f}"
        for i, hyp in enumerate(hypotheses)
    ]
    return {
        "hypotheses":      hypotheses,
        "prior_probs":     P_prior.tolist(),
        "posterior_probs": P_post.tolist(),
        "KL_divergence":   kl,
        "JS_divergence": js_norm,
    }



def run_bayesian_surprise_over_video(video_frames, window_size, num_hypotheses, method="prior_frame_bayesian_approach", model=None, processor=None):
    """
    Runs Bayesian surprise over a sequence of video frames.
    Args:
        video_frames: List of video frames (PIL images).
        window_size: Number of frames to consider for context.
        num_hypothesis: Number of hypotheses to generate."""
    surprise_scores = []
    running_memory = ""
    priors = []
    posteriors = []
    hypotheses = []

    for i in tqdm(range(window_size, len(video_frames), 1), desc="Processing frames"):
        prior_window = video_frames[i-window_size:i]
        observed_frame = video_frames[i]
        running_memory = summarize_memory(running_memory)
        if method == "prior_frame_bayesian_approach":
            result = qwen_bayesian_surprise_text_future(
                memory_text=running_memory,
                context_frames=prior_window,
                observed_frame=observed_frame,
                num_hypotheses=num_hypotheses,
                model=model,
                processor=processor
            )
        caption = caption_frame(
            context_frames=prior_window,
            observed_frame=observed_frame,
            memory_text=running_memory,
            model=model,
            processor=processor
        )
        result["caption"] = caption
        running_memory += (f" Then, {caption}")
        surprise_scores.append(result["JS_divergence"])
        priors.append(result["prior_probs"])
        posteriors.append(result["posterior_probs"])
        hypotheses.append(result["hypotheses"])
        print(f"############################ Finished processing frame {i} ############################")

    # Pad 0s for the first few frames where we don't have enough context
    for _ in range(window_size):
        surprise_scores.insert(0, 0.0)
        priors.insert(0, [0.0] * num_hypotheses)
        posteriors.insert(0, [0.0] * num_hypotheses)
        hypotheses.insert(0, [""] * num_hypotheses)
    # Convert list inside hypotheses into single string with \n between them dd

    return {
        "surprise_scores": surprise_scores,
        "memory_evolution": running_memory,
        "priors": priors,
        "posteriors": posteriors,
        "explanations": hypotheses,
        "frames": video_frames,
    }


def qwen_surprise_tracker(
    frames: List[np.ndarray],
    window_size: int = 4,
    top_k: int = 3,
    method: str = "prior_frame_bayesian_approach",
    caption_video: bool = True,
    vr=None,
    model=None,
    processor=None
) -> Dict[str, List]:
    """
    Process the video to extract frames and save them to the output directory.
    """

    print(f"Using method: {method}")
    surprise_output = run_bayesian_surprise_over_video(
        video_frames=frames,
        window_size=window_size,
        num_hypotheses=top_k,
        method=method,
        model=model,
        processor=processor

    )

    if caption_video:
        caption_weighted, sampled_frames_weighted = caption_by_weight(
            model=model,
            processor=processor,
            frames=frames,
            scores=surprise_output["surprise_scores"],
            vr=vr
           
        )
        caption_unweighted, sampled_frames_unweighted = caption_by_weight(
            model=model,
            processor=processor,
            frames=frames,
            scores=[1.0] * len(frames),
            vr = vr
     
        )
        print("caption_weighted", caption_weighted)
        print("caption_unweighted", caption_unweighted)
        surprise_output["caption_weighted"] = caption_weighted
        surprise_output["caption_unweighted"] = caption_unweighted
        surprise_output["sampled_frames_weighted"] = sampled_frames_weighted
        surprise_output["sampled_frames_unweighted"] = sampled_frames_unweighted
    return surprise_output 

    




