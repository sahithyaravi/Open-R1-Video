from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# model_path = "Qwen/Qwen2-VL-7B-Instruct"
model_path = "/home/sahiravi/projects/aip-vshwartz/sahiravi/Open-R1-Video/data/ckpt/Qwen2-VL-7B-Video-GRPO/llava-video-4k-remove-formatreward-matchletterreward-f16-full/checkpoint-1100"

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_path)

# Messages containing a local video path and a text query

# query = """
# What does the person do with the white powdery substance from the larger bowl?
# A. They sprinkle it over the countertop
# B. They pour it into the food processor
# C. They mix it with a liquid
# D. They use a spoon to add it to the food processor
# Answer with the option's letter from the given choices directly.\n
# """
# query = """
# What does the person do with the white powdery substance from the larger bowl?
# A. They sprinkle it over the countertop
# B. They pour it into the food processor
# C. They mix it with a liquid
# D. They use a spoon to add it to the food processor
# Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.
# """


# query = """
# What does the person do with the white powdery substance from the larger bowl?
# A. They sprinkle it over the countertop
# B. They pour it into the food processor
# C. They mix it with a liquid
# D. They use a spoon to add it to the food processor
# Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.
# """


video_path = "/home/sahiravi/scratch/oops/oops_val_v1_merged/13_D_merged.mp4"
query = """
Describe this video in detail.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.
"""




# Monkey video
# video_path = "/home/sahiravi/scratch/oops/oops_val_v1_merged/52_E_merged.mp4"
# query = """
# Given this video, which of these hypothesis are valid?
# A. The monkey enters the van through the window, grabs a bag and runs away with the bag, after slapping the driver of the van.
# B. The monkey enters the van through the window, grabs a bag and runs away with the bag, while the driver laughs at it. 
# C. The monkey enters the van through the window, grabs a bag and runs away with the bag, while the driver tries to stop it. 
# Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.
# """

# query = """
# Describe this video in detail.
# Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.
# """


# assets/split_5.mp4
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "text": f"{video_path}",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": query},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    videos=video_inputs,
    fps=1,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)