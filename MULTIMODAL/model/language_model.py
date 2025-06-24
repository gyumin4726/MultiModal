import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 캐시 디렉토리 생성
os.makedirs("./model_cache", exist_ok=True)


class MultimodalLanguageModel(nn.Module):
    """BLIP2 스타일 멀티모달 언어 모델 - OPT 기반"""
    
    def __init__(self, model_name="facebook/opt-2.7b", device=None):
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 토크나이저 로딩
        print(f"Loading multimodal tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./model_cache",
            local_files_only=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # OPT 언어 모델 로딩
        print(f"Loading multimodal language model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            cache_dir="./model_cache",
            local_files_only=False,
            trust_remote_code=True
        )
        self.model.to(self.device)
        
        # 🔥 핵심: Vision Token Projection Layer
        # Q-Former 출력(768D)을 LLM 임베딩 차원으로 변환
        self.vision_token_proj = nn.Linear(768, self.model.config.hidden_size).to(self.device)
        
        print(f"✅ Multimodal LLM loaded successfully on {self.device}")
        print(f"✅ Vision token projection: 768 → {self.model.config.hidden_size}")
    
    def forward_with_vision_tokens(self, vision_tokens, text_input_ids, attention_mask=None):
        """비전 토큰과 텍스트를 함께 처리 - BLIP2 방식"""
        
        batch_size = text_input_ids.shape[0]
        
        # 1. 비전 토큰을 LLM 임베딩 차원으로 변환
        vision_embeddings = self.vision_token_proj(vision_tokens)  # [1, 32, hidden_size]
        
        # 2. 텍스트 토큰을 임베딩으로 변환
        text_embeddings = self.model.get_input_embeddings()(text_input_ids)  # [1, seq_len, hidden_size]
        
        # 3. 비전 토큰과 텍스트 토큰을 결합
        # [Vision Tokens] + [Text Tokens] 순서로 결합
        combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)
        
        # 4. Attention mask도 확장
        if attention_mask is not None:
            vision_attention_mask = torch.ones(batch_size, vision_tokens.shape[1], device=self.device)
            combined_attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)
        else:
            combined_attention_mask = None
        
        # 5. LLM forward pass (임베딩을 직접 입력)
        outputs = self.model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            use_cache=False
        )
        
        return outputs
    
    def generate_with_vision(self, vision_tokens, text_prompt, max_new_tokens=50, temperature=0.0):
        """비전 토큰과 함께 텍스트 생성"""
        
        # 텍스트 토크나이징
        text_inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        
        batch_size = text_inputs['input_ids'].shape[0]
        
        # 비전 토큰을 임베딩으로 변환
        vision_embeddings = self.vision_token_proj(vision_tokens)  # [1, 32, hidden_size]
        
        # 텍스트 임베딩
        text_embeddings = self.model.get_input_embeddings()(text_inputs['input_ids'])
        
        # 결합된 임베딩
        combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)
        
        # Attention mask 생성
        vision_attention_mask = torch.ones(batch_size, vision_tokens.shape[1], device=self.device)
        combined_attention_mask = torch.cat([vision_attention_mask, text_inputs['attention_mask']], dim=1)
        
        # 생성 설정
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        
        if temperature > 0:
            generation_config.update({
                "temperature": temperature,
                "top_p": 0.9
            })
        
        # 🔥 핵심: 임베딩을 사용한 생성
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                **generation_config
            )
        
        # 생성된 부분만 디코딩 (원본 입력 제외)
        generated_tokens = outputs[0][combined_embeddings.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def answer_vqa_multimodal(self, vision_tokens, question, choices):
        """진짜 멀티모달 VQA - 비전 토큰을 직접 처리"""
        
        # 선택지 텍스트 생성
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # 명확한 형식 지정 프롬프트
        prompt = f"""Based on the image, answer the following question.

Question: {question}

Options:
{choices_text}

Instructions: Choose the best answer from the options above. Respond with ONLY the letter (A, B, C, or D) followed by a period.

Answer:"""
        
        # 비전 토큰과 함께 생성
        response = self.generate_with_vision(
            vision_tokens=vision_tokens,
            text_prompt=prompt,
            max_new_tokens=5,  # 더 짧게 설정 (A. 형식만 필요)
            temperature=0.0
        )
        
        return response


def load_language_model(**kwargs):
    """언어 모델 로딩 함수 - BLIP2 스타일만 지원"""
    return MultimodalLanguageModel(**kwargs) 