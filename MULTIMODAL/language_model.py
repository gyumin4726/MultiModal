import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


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
    
    def describe_visual_features(self, vision_features):
        """비전 피처를 텍스트 설명으로 변환"""
        if vision_features is None:
            return "No visual information available."
        
        # 피처를 numpy로 변환
        if isinstance(vision_features, torch.Tensor):
            features = vision_features.detach().cpu().numpy()
            if features.ndim > 1:
                features = features.flatten()
        else:
            features = np.array(vision_features).flatten()
        
        # 피처 통계 분석
        mean_val = np.mean(features)
        std_val = np.std(features)
        max_val = np.max(features)
        min_val = np.min(features)
        
        # 주요 피처 패턴 분석
        high_activation_ratio = np.sum(features > mean_val + std_val) / len(features)
        low_activation_ratio = np.sum(features < mean_val - std_val) / len(features)
        
        # 시각적 특성 추론
        visual_description = []
        
        # 활성화 패턴 기반 추론
        if high_activation_ratio > 0.15:
            visual_description.append("complex visual patterns detected")
        if low_activation_ratio > 0.15:
            visual_description.append("simple or uniform areas present")
        
        # 피처 강도 기반 추론
        if max_val > 2.0:
            visual_description.append("strong visual features")
        elif max_val < 0.5:
            visual_description.append("subtle visual features")
        
        # 피처 분산 기반 추론
        if std_val > 1.0:
            visual_description.append("diverse visual elements")
        elif std_val < 0.3:
            visual_description.append("uniform visual appearance")
        
        if not visual_description:
            visual_description = ["moderate visual complexity"]
        
        return f"Visual analysis: {', '.join(visual_description)} (feature stats: mean={mean_val:.2f}, std={std_val:.2f})"
    
    def describe_text_features(self, text_features):
        """텍스트 피처를 설명으로 변환"""
        if text_features is None:
            return "No text feature analysis available."
        
        # 피처를 numpy로 변환
        if isinstance(text_features, torch.Tensor):
            features = text_features.detach().cpu().numpy()
            if features.ndim > 1:
                features = features.flatten()
        else:
            features = np.array(text_features).flatten()
        
        # 텍스트 피처 분석
        mean_val = np.mean(features)
        std_val = np.std(features)
        
        semantic_strength = "strong" if std_val > 0.5 else "moderate" if std_val > 0.2 else "weak"
        
        return f"Text semantic analysis: {semantic_strength} semantic encoding (mean={mean_val:.2f}, std={std_val:.2f})"
    
    def answer_question_with_features(self, question, choices, vision_features=None, text_features=None, fused_features=None):
        """피처를 활용한 VQA 추론"""
        
        # 선택지 포맷팅
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # 기본 프롬프트
        prompt_parts = [
            "You are an expert AI that analyzes images and answers questions.",
            "Use the provided visual and semantic information to select the best answer.\n"
        ]
        
        # 피처 정보 추가
        if vision_features is not None:
            visual_desc = self.describe_visual_features(vision_features)
            prompt_parts.append(f"VISUAL INFORMATION: {visual_desc}\n")
        
        if text_features is not None:
            text_desc = self.describe_text_features(text_features)
            prompt_parts.append(f"SEMANTIC INFORMATION: {text_desc}\n")
        
        if fused_features is not None:
            fused_desc = self.describe_visual_features(fused_features)  # 같은 방식으로 분석
            prompt_parts.append(f"MULTIMODAL ANALYSIS: Combined visual-semantic features show {fused_desc.split(': ')[1]}\n")
        
        # 질문과 선택지 추가
        prompt_parts.extend([
            f"QUESTION: {question}",
            f"CHOICES:\n{choices_text}",
            "",
            "Based on the visual and semantic analysis above, select the most appropriate answer.",
            "Consider both the visual features and the question context.",
            "Answer with just the letter (A, B, C, or D):",
            ""
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # LLM 추론
        try:
            response = self.generate_text(prompt, max_new_tokens=5, temperature=0.0)
            return response.strip()
        except Exception as e:
            print(f"Warning: Feature-based inference failed - {e}")
            return self.answer_question_simple(question, choices)
    
    def answer_question_simple(self, question, choices):
        """간단한 텍스트 전용 VQA (fallback)"""
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        prompt = (
            "You are a helpful AI that answers multiple-choice questions.\n"
            "Select the best answer from A, B, C, or D.\n\n"
            f"Question: {question}\n"
            f"{choices_text}\n\n"
            "Answer:"
        )
        
        try:
            response = self.generate_text(prompt, max_new_tokens=5, temperature=0.0)
            return response.strip()
        except Exception as e:
            print(f"Warning: Simple inference failed - {e}")
            return "A"
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def load_language_model(**kwargs):
    """언어 모델 로딩 함수"""
    return LanguageModel(**kwargs) 