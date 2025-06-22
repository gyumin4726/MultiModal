import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel(nn.Module):
    """microsoft/phi-2 언어 모델 (생성용) - PyTorch 1.12.1 호환"""
    
    def __init__(self, model_name="microsoft/phi-2", device=None):
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 토크나이저 로딩
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # LLM 모델 로딩 (PyTorch 1.12.1 호환)
        print(f"Loading language model: {model_name}")
        try:
            # PyTorch 1.12.1에서는 device_map="auto"가 지원되지 않을 수 있음
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Failed to load {model_name} with float16, trying float32...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device} with float32")
        
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.0):
        """텍스트 생성"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # PyTorch 1.12.1 호환 설정
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if temperature > 0:
                generation_config.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": 0.9
                })
            else:
                generation_config.update({
                    "do_sample": False,
                    "num_beams": 1
                })
            
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 원래 입력을 제외하고 새로 생성된 부분만 반환
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return generated_text.strip()
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def load_language_model(**kwargs):
    """언어 모델 로딩 함수"""
    return LanguageModel(**kwargs) 