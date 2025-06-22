import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel(nn.Module):
    """microsoft/phi-2 언어 모델 (생성용)"""
    
    def __init__(self, model_name="microsoft/phi-2"):
        super().__init__()
        
        self.model_name = model_name
        
        # 토크나이저 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # LLM 모델 로딩
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.0):
        """텍스트 생성"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id
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