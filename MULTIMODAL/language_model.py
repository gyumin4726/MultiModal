import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

# 캐시 디렉토리 생성
os.makedirs("./model_cache", exist_ok=True)


class LanguageModel(nn.Module):
    """microsoft/phi-2 언어 모델 (안전한 텍스트 기반) - PyTorch 1.12.1 호환"""
    
    def __init__(self, model_name="microsoft/phi-2", device=None):
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 토크나이저 로딩 (캐시 활용)
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./model_cache",
            local_files_only=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # LLM 모델 로딩 (PyTorch 1.12.1 호환 + 캐시 최적화)
        print(f"Loading language model: {model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True,
                cache_dir="./model_cache",
                local_files_only=False,
                trust_remote_code=True
            )
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device} with float16")
        except Exception as e:
            print(f"Failed to load {model_name} with float16, trying float32...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                cache_dir="./model_cache",
                local_files_only=False,
                trust_remote_code=True
            )
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device} with float32")
        
        print("✅ Safe text-based language model initialized!")
        
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.0):
        """안전한 텍스트 생성"""
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
    
    def analyze_visual_features_advanced(self, vision_features):
        """고급 비전 피처 분석"""
        if vision_features is None:
            return "No visual information available."
        
        # 피처를 numpy로 변환
        if isinstance(vision_features, torch.Tensor):
            features = vision_features.detach().cpu().numpy()
            if features.ndim > 1:
                features = features.flatten()
        else:
            features = np.array(vision_features).flatten()
        
        # 고급 통계 분석
        mean_val = np.mean(features)
        std_val = np.std(features)
        max_val = np.max(features)
        min_val = np.min(features)
        median_val = np.median(features)
        
        # 분포 분석
        q25 = np.percentile(features, 25)
        q75 = np.percentile(features, 75)
        iqr = q75 - q25
        
        # 활성화 패턴 분석
        high_activation = np.sum(features > mean_val + std_val) / len(features)
        low_activation = np.sum(features < mean_val - std_val) / len(features)
        zero_activation = np.sum(np.abs(features) < 0.01) / len(features)
        
        # 시각적 복잡도 추정
        complexity_score = std_val / (abs(mean_val) + 1e-6)
        
        # 시각적 특성 추론
        visual_insights = []
        
        # 복잡도 기반 분석
        if complexity_score > 2.0:
            visual_insights.append("highly complex visual scene")
        elif complexity_score > 1.0:
            visual_insights.append("moderately complex visual content")
        else:
            visual_insights.append("simple visual structure")
        
        # 활성화 패턴 기반 분석
        if high_activation > 0.2:
            visual_insights.append("strong distinctive features present")
        if low_activation > 0.2:
            visual_insights.append("background or uniform areas detected")
        if zero_activation > 0.3:
            visual_insights.append("sparse feature representation")
        
        # 강도 기반 분석
        if max_val > 3.0:
            visual_insights.append("prominent visual elements")
        elif max_val < 0.5:
            visual_insights.append("subtle visual details")
        
        # 분산 기반 분석
        if iqr > 1.5:
            visual_insights.append("diverse visual components")
        elif iqr < 0.3:
            visual_insights.append("consistent visual appearance")
        
        if not visual_insights:
            visual_insights = ["standard visual characteristics"]
        
        return f"Visual analysis: {', '.join(visual_insights)} (complexity: {complexity_score:.2f}, activation: {high_activation:.2f})"
    
    def analyze_text_features_advanced(self, text_features):
        """고급 텍스트 피처 분석"""
        if text_features is None:
            return "No text semantic analysis available."
        
        # 피처를 numpy로 변환
        if isinstance(text_features, torch.Tensor):
            features = text_features.detach().cpu().numpy()
            if features.ndim > 1:
                features = features.flatten()
        else:
            features = np.array(text_features).flatten()
        
        # 고급 의미 분석
        mean_val = np.mean(features)
        std_val = np.std(features)
        max_val = np.max(features)
        min_val = np.min(features)
        
        # 의미적 강도 분석
        semantic_strength = std_val
        semantic_focus = abs(mean_val)
        semantic_range = max_val - min_val
        
        # 의미적 특성 추론
        semantic_insights = []
        
        if semantic_strength > 0.8:
            semantic_insights.append("rich semantic content")
        elif semantic_strength > 0.4:
            semantic_insights.append("moderate semantic complexity")
        else:
            semantic_insights.append("simple semantic structure")
        
        if semantic_focus > 0.5:
            semantic_insights.append("focused conceptual meaning")
        elif semantic_focus < 0.1:
            semantic_insights.append("balanced semantic distribution")
        
        if semantic_range > 2.0:
            semantic_insights.append("diverse conceptual elements")
        
        if not semantic_insights:
            semantic_insights = ["standard semantic encoding"]
        
        return f"Semantic analysis: {', '.join(semantic_insights)} (strength: {semantic_strength:.2f}, focus: {semantic_focus:.2f})"
    
    def analyze_fused_features_advanced(self, fused_features):
        """고급 융합 피처 분석"""
        if fused_features is None:
            return "No multimodal fusion analysis available."
        
        # 피처를 numpy로 변환
        if isinstance(fused_features, torch.Tensor):
            features = fused_features.detach().cpu().numpy()
            if features.ndim > 1:
                features = features.flatten()
        else:
            features = np.array(fused_features).flatten()
        
        # 융합 품질 분석
        mean_val = np.mean(features)
        std_val = np.std(features)
        
        # 융합 특성 추론
        fusion_quality = std_val / (abs(mean_val) + 1e-6)
        
        fusion_insights = []
        
        if fusion_quality > 1.5:
            fusion_insights.append("strong multimodal alignment")
        elif fusion_quality > 0.8:
            fusion_insights.append("moderate cross-modal integration")
        else:
            fusion_insights.append("basic multimodal combination")
        
        if abs(mean_val) > 0.5:
            fusion_insights.append("decisive multimodal representation")
        
        return f"Multimodal fusion: {', '.join(fusion_insights)} (quality: {fusion_quality:.2f})"
    
    def answer_question_with_features(self, question, choices, vision_features=None, text_features=None, fused_features=None):
        """고급 피처 분석 기반 VQA 추론"""
        
        # 선택지 포맷팅
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # 간결한 프롬프트 구성 (토큰 절약)
        analysis_parts = []
        
        # 각 피처 타입별 간결한 분석 추가
        if vision_features is not None:
            visual_analysis = self.analyze_visual_features_advanced(vision_features)
            analysis_parts.append(f"Visual: {visual_analysis}")
        
        if text_features is not None:
            text_analysis = self.analyze_text_features_advanced(text_features)
            analysis_parts.append(f"Semantic: {text_analysis}")
        
        if fused_features is not None:
            fusion_analysis = self.analyze_fused_features_advanced(fused_features)
            analysis_parts.append(f"Fusion: {fusion_analysis}")
        
        # 간결한 프롬프트 구성
        if analysis_parts:
            feature_info = "\n".join(analysis_parts)
            prompt = (
                f"Analysis:\n{feature_info}\n\n"
                f"Question: {question}\n"
                f"{choices_text}\n\n"
                f"Answer:"
            )
        else:
            # 피처가 없으면 더 간단한 프롬프트
            prompt = (
                f"Question: {question}\n"
                f"{choices_text}\n\n"
                f"Answer:"
            )
        
        # 안전한 텍스트 생성 (토큰 수 증가)
        try:
            response = self.generate_text(prompt, max_new_tokens=20, temperature=0.0)
            return response.strip()
        except Exception as e:
            print(f"Warning: Advanced feature-based inference failed - {e}")
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