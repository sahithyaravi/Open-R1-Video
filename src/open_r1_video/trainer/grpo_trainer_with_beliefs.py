# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import gc
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from sentence_transformers import SentenceTransformer, util
from qwen_vl_utils import process_vision_info
from .belief_tracker import qwen_surprise_tracker
from .weighted_captioning import adaptive_frame_sampling
from .video_processing import extract_k_frames_decord_cpu
from typing import List
from torch.nn import functional as F
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention
VisionAttention.is_causal=False
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainerBelief(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model_init_kwargs["torch_dtype"] = torch.bfloat16
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen2-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.sim_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            # if self.is_deepspeed_enabled:
            #     self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            # else:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def _simple_cosine_similarity(self, text1: str, text2: str) -> float:
        # 1. Encode each sentence into an embedding (vector)
        emb1 = self.sim_encoder.encode(text1, convert_to_tensor=True)
        emb2 = self.sim_encoder.encode(text2, convert_to_tensor=True)

        # 2. Compute cosine similarity between the two embeddings
        similarity_value = util.cos_sim(emb1, emb2)

        return similarity_value

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("This customised compute_loss does not support returning outputs")

        
        total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        m = 2
        beta = 0.02

        for example in inputs:
            video_path = example.get("video") or example.get("video_path")
            ground_truth = example.get("ground_truth")

            # ---- Surprise & selection OUTSIDE grad ----
            with torch.no_grad():
                frames, frame_indices, total_frames, fps, vr = extract_k_frames_decord_cpu(
                    video_path=video_path, k=8
                )
                # <<< CHANGED >>> don’t generate inside the tracker
                result = qwen_surprise_tracker(
                    frames=frames,
                    window_size=4,
                    top_k=3,
                    method="prior_frame_bayesian_approach",
                    caption_video=False,   # <<< CHANGED
                    vr=vr,
                    model=model,
                    processor=self.processing_class,
                )
                top_frames, frame_indices = adaptive_frame_sampling(
                    scores=result["surprise_scores"], vr=vr
                )
                print("Surprise scores:", result["surprise_scores"])

            # ---- Build caption prompt once (uses top_frames) ----
            cap_conv = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the key events in the video in one sentence."},
                    {"type": "video"},
                ],
            }]
            cap_prompt = self.processing_class.apply_chat_template(cap_conv, add_generation_prompt=True)

            # We’ll sample m captions, then score them in one batched pass
            group_rewards = []
            caption_texts = []          # <<< CHANGED
            # (We will fill log-probs later in a batched pass)
            # group_logps, group_logps_ref removed here

            # -------- [C] SAMPLE m captions (no grad) --------
            for _ in range(m):
                with torch.no_grad():
                    gen_inputs = self.processing_class(
                        text=cap_prompt, videos=top_frames, return_tensors="pt"
                    ).to(model.device)
                    gen_out = model.generate(
                        **gen_inputs,
                        max_new_tokens=50,
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        return_dict_in_generate=True,
                    )
                    generated_ids = gen_out.sequences[0]
                    caption_text = self.processing_class.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                # reward only on final caption
                reward = self._simple_cosine_similarity(caption_text, str(ground_truth))
                group_rewards.append(float(reward))
                caption_texts.append(caption_text)
                      # Get prompt length once
            with torch.no_grad():
                prompt_inputs = self.processing_class(
                    text=cap_prompt, videos=top_frames, return_tensors="pt"
                ).to(model.device)
                prompt_len = prompt_inputs["input_ids"].shape[1]

            sum_logp_list = []
            for cap in caption_texts:
                # Build the full conversation for this single caption
                cap_conv_scored = [
                    {"role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize the key events in the video in one sentence."},
                        {"type": "video"},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": cap}]},
                ]
                full_text = self.processing_class.apply_chat_template(
                    cap_conv_scored, add_generation_prompt=False
                )

                # Tokenize the single example
                scored_inputs = self.processing_class(
                    text=full_text,
                    videos=top_frames, # Pass video frames for just this one sample
                    return_tensors="pt",
                ).to(model.device)

                input_ids = scored_inputs["input_ids"]
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100 # Mask prompt tokens

                # Policy forward pass for a single caption
                out = model(**scored_inputs)
                logits = out.logits.float()

                # Calculate cross-entropy loss for the tokens of this caption
                ce = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    reduction="none",
                )
                
                # Reshape and mask to get per-token loss for the sequence
                ce = ce.view(1, -1) # Reshape to [1, L-1]
                token_mask = (labels[:, 1:] != -100).float()
                ce = ce * token_mask

                # Sum the log-probabilities for this caption and append to list
                sum_logp_list.append(-(ce.sum()))

                # Explicitly delete tensors to free memory
                del scored_inputs, out, logits, ce, token_mask
                torch.cuda.empty_cache()
                gc.collect()


            sum_logp = torch.stack(sum_logp_list) # shape [m], requires_grad=True
            print("Cross entropy losses calculated serially.", sum_logp)

            # Reference forward (no grad), batched
            # ----- SIMPLE REFERENCE SCORING (no slicing) -----
            sum_logp_ref_list = []
            ref_device = next(self.ref_model.parameters()).device  # e.g., "cuda:1" or "cpu"

            with torch.inference_mode():
                # prompt_len computed earlier from cap_prompt + top_frames
                for cap in caption_texts:
                    # Rebuild full text for THIS caption
                    conv = [
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": "Summarize the key events in the video in one sentence."},
                            {"type": "video"},
                        ]},
                        {"role": "assistant", "content": [{"type": "text", "text": cap}]},
                    ]
                    full_text = self.processing_class.apply_chat_template(
                        conv, add_generation_prompt=False
                    )

                    # Re-tokenize for this single sample
                    mb = self.processing_class(text=full_text, videos=top_frames, return_tensors="pt")
                    labels_mb = mb["input_ids"].clone()
                    labels_mb[:, :prompt_len] = -100  # score continuation only

                    mb = {k: v.to(ref_device) for k, v in mb.items()}
                    labels_mb = labels_mb.to(ref_device)

                    # bf16 autocast if on GPU
                    ctx = torch.autocast("cuda", dtype=torch.bfloat16)
                    with ctx:
                        out_ref = self.ref_model(**mb, labels=labels_mb)  # mean CE over labeled tokens
                        num_lab = (labels_mb != -100).sum()
                        sum_logp_ref_i = -(out_ref.loss.float() * num_lab)

                    sum_logp_ref_list.append(sum_logp_ref_i.to(model.device))


                    del mb, labels_mb, out_ref
                    torch.cuda.empty_cache()
                    gc.collect()

            sum_logp_ref = torch.stack(sum_logp_ref_list)  # shape [m]

            # -------- [G] advantages (float32) --------
            rewards = torch.tensor(group_rewards, device=model.device, dtype=torch.float32)
            adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).detach()

            # -------- [H] GRPO loss --------
            loss_ex = -torch.mean(adv * (sum_logp - beta * sum_logp_ref))
            total_loss = total_loss + loss_ex
            print(f"Loss for this example: {loss_ex.item()}")
            torch.cuda.empty_cache()
            gc.collect()

        if len(inputs) == 0:
            return total_loss # Return 0 if batch is empty
        return total_loss / len(inputs)
        
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))