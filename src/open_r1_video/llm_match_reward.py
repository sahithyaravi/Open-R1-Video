import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Union

class HuggingFaceLLMReward:
    """Lightweight LLM-based reward function using OLMo or Qwen models"""
    
    def __init__(self, 
                 model_name: str = "allenai/OLMo-1B-hf",  # Default to OLMo 1B
                 device: str = "cuda",
                 max_length: int = 512):
        """
        Initialize with lightweight models:
        
        OLMo options:
        - "allenai/OLMo-1B-hf" (1B parameters, very fast)
        - "allenai/OLMo-7B-hf" (7B parameters, better quality)
        
        Qwen options:
        - "Qwen/Qwen2-0.5B" (0.5B parameters, ultra lightweight)
        - "Qwen/Qwen2-1.5B" (1.5B parameters, good balance)
        - "Qwen/Qwen2-7B" (7B parameters, high quality)
        - "Qwen/Qwen1.5-0.5B-Chat" (0.5B, chat optimized)
        
        Recommended: "Qwen/Qwen2-0.5B" for speed, "allenai/OLMo-1B-hf" for balance
        """
        
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        print(f"Loading lightweight LLM reward model: {model_name}")
        
        # Load model and tokenizer with optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if not torch.cuda.is_available():
            self.model.to(self.device)
        self.model.eval()
        
        # Handle different tokenizer configurations
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        print(f"Model loaded on {self.device}, parameters: ~{self._count_parameters():.1f}M")
    
    @torch.no_grad()
    def __call__(self, responses: List[str], ground_truths: List[str]) -> torch.Tensor:
        """Compute rewards for responses vs ground truths"""
        rewards = []
        
        for response, gt in zip(responses, ground_truths):
            prompt = f"""Rate how well the response matches the expected answer (0.0-1.0):

Expected: {gt}
Response: {response}

Score:"""
            
            try:
                # Tokenize and generate
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Extract score
                generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                score = self._extract_score(generated)
                rewards.append(score)
                
            except Exception as e:
                print(f"LLM reward error: {e}")
                rewards.append(0.5)  # fallback
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _extract_score(self, text: str) -> float:
        """Extract numerical score from generated text"""
        match = re.search(r'(\d+\.?\d*)', text.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score / 100.0 if score > 1.0 else score))
        return 0.5