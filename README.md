# Open R1 Video

We introduce R1's paradigm to video understanding tasks and open-sourced the training code and data.

[🤗 Datasets](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k) | [Wandb Logs](https://wandb.ai/xiaodongwang/Qwen-2-VL-7B-Video-GRPO/workspace?nw=nwuserxiaodongwang)

> [!NOTE] 
> Although our insights may not be guaranteed to be correct, we commit to sharing them truthfully and honestly. We welcome community feedback and discussions to improve our understanding on multimodal reasoning models.

## News
- [2025/02/18] We release training code and data of Open-R1-Video!

## Our Findings
### GRPO training that forces thinking can improve video understanding
We train [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) on simple video dataset [open-r1-video-4k](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k) using 4 x A100 (80G) GPUs, and the training only utilize video, query, and the ground truth answer (the letter of the correct answer). We only used GRPO (pure reinforcement learning without labeled reasoning trajectories) to train the model and achieved considerable rewards during model training. We release our [wandb logs](https://wandb.ai/xiaodongwang/Qwen-2-VL-7B-Video-GRPO/workspace?nw=nwuserxiaodongwang) for reference.
![image](assets\log.png)

**What We Did**
- Introduce R1 to Video-LMM (e.g., Qwen2-VL) based on [huggingface/open-r1](https://github.com/huggingface/open-r1) and [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1). 
- Open-sourced the simple training data [open-r1-video-4k](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k).
  - The simple reformat data is available in [open-r1-video-4k](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k).
  - The video data is available in [LLaVA-Video-large-swift](https://huggingface.co/datasets/malterei/LLaVA-Video-large-swift).



## Training Models

> [!NOTE]
> The training commands below are configured for a node of 4 x A100 (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

### Set up
```
git clone https://github.com/Wang-Xiaodong1899/Open-R1-Video.git
cd Open-R1-Video
conda create -n r1 python=3.10
conda activate r1
pip3 install -e ".[dev]"
pip3 install flash_attn --no-build-isolation
cd qwen-vl-utils
pip install -e .
cd ..

# download data and put in data/
wget https://huggingface.co/datasets/Xiaodong/open-r1-video-4k/resolve/main/LLaVA-Video-large-swift-origin.jsonl
# like: data/LLaVA-Video-large-swift-origin.jsonl

# download videos
git lfs install
git clone https://huggingface.co/datasets/malterei/LLaVA-Video-large-swift

```


### GRPO on Qwen2-VL/7B

To run GRPO on Qwen2-VL-7B:

```
bash qwen-7b.sh
```

Please refer to [qwen-7b.sh](qwen-7b.sh) for more details.


### Evaluating models
On-going...

### RL Data Reformat

We provide the easy reformat method to obtain the data for GRPO training, which only utilize video, query, and final answer. Please refer to [format_video_data.py](scripts\format_video_data.py) for more details.

Users can view data in [open-r1-video-4k](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k). The `original question`/`original answer` are from the original dataset.

## References & Acknowledgements
We sincerely thank the contributions from the open source community, including the reproduction of [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), and [R1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), etc.

The related projects are as follows:
- [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)
- [lmm-r1](https://github.com/TideDra/lmm-r1)
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) 
- [open-r1](https://github.com/huggingface/open-r1)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [LLaVA-Video-large-swift](https://huggingface.co/datasets/malterei/LLaVA-Video-large-swift)