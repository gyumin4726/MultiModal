import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

# 캐시 디렉토리 생성
os.makedirs("./model_cache", exist_ok=True)


class LanguageModel(nn.Module):
    """microsoft/phi-2 언어 모델 - VQA 특화"""
    
    def __init__(self, model_name="microsoft/phi-2", device=None):
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 토크나이저 로딩
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./model_cache",
            local_files_only=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # LLM 모델 로딩
        print(f"Loading language model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            cache_dir="./model_cache",
            local_files_only=False,
            trust_remote_code=True
        )
        self.model.to(self.device)
        print(f"✅ Model loaded successfully on {self.device}")
        
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.0):
        """텍스트 생성"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
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
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return generated_text.strip()
    
    def analyze_visual_features_advanced(self, vision_features):
        """고급 비전 피처 분석"""
        if vision_features is None:
            return "No visual information available."
        
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
        
        if complexity_score > 2.0:
            visual_insights.append("highly complex visual scene")
        elif complexity_score > 1.0:
            visual_insights.append("moderately complex visual content")
        else:
            visual_insights.append("simple visual structure")
        
        if high_activation > 0.2:
            visual_insights.append("strong distinctive features present")
        if low_activation > 0.2:
            visual_insights.append("background or uniform areas detected")
        if zero_activation > 0.3:
            visual_insights.append("sparse feature representation")
        
        if max_val > 3.0:
            visual_insights.append("prominent visual elements")
        elif max_val < 0.5:
            visual_insights.append("subtle visual details")
        
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
            semantic_insights.append("broad semantic coverage")
        
        if not semantic_insights:
            semantic_insights = ["standard semantic characteristics"]
        
        return f"Text analysis: {', '.join(semantic_insights)} (strength: {semantic_strength:.2f}, focus: {semantic_focus:.2f})"
    
    def analyze_fused_features_advanced(self, fused_features):
        """고급 융합 피처 분석"""
        if fused_features is None:
            return "No multimodal fusion analysis available."
        
        if isinstance(fused_features, torch.Tensor):
            features = fused_features.detach().cpu().numpy()
            if features.ndim > 1:
                features = features.flatten()
        else:
            features = np.array(fused_features).flatten()
        
        # 융합 품질 분석
        mean_val = np.mean(features)
        std_val = np.std(features)
        max_val = np.max(features)
        
        # 융합 효과성 측정
        fusion_strength = std_val
        fusion_balance = 1.0 - abs(mean_val)  # 0에 가까울수록 균형적
        
        fusion_insights = []
        
        if fusion_strength > 1.0:
            fusion_insights.append("strong multimodal integration")
        elif fusion_strength > 0.5:
            fusion_insights.append("moderate cross-modal alignment")
        else:
            fusion_insights.append("basic multimodal combination")
        
        if fusion_balance > 0.8:
            fusion_insights.append("well-balanced modality fusion")
        elif fusion_balance < 0.3:
            fusion_insights.append("modality-biased representation")
        
        if not fusion_insights:
            fusion_insights = ["standard multimodal fusion"]
        
        return f"Fusion analysis: {', '.join(fusion_insights)} (strength: {fusion_strength:.2f}, balance: {fusion_balance:.2f})"
    
    def answer_question_with_features(self, question, choices, vision_features=None, text_features=None, fused_features=None):
        """피처 기반 VQA 추론"""
        
        # 피처 분석
        feature_analyses = []
        if vision_features is not None:
            feature_analyses.append(self.analyze_visual_features_advanced(vision_features))
        if text_features is not None:
            feature_analyses.append(self.analyze_text_features_advanced(text_features))
        if fused_features is not None:
            feature_analyses.append(self.analyze_fused_features_advanced(fused_features))
        
        feature_info = "\n".join(feature_analyses) if feature_analyses else "No feature analysis available."
        
        # 선택지 텍스트 생성
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # VQA 프롬프트 생성
        prompt = self.create_advanced_vqa_prompt(question, choices_text, feature_info)
        
        # 답변 생성
        response = self.generate_text(prompt, max_new_tokens=200, temperature=0.0)
        
        # 자기 검증
        final_answer = self.extract_final_answer(response)
        verified_response = self.self_verify_answer(question, choices_text, feature_info, final_answer, response)
        
        return verified_response
    
    def create_advanced_vqa_prompt(self, question, choices_text, feature_info):
        """고급 VQA 프롬프트 생성"""
        
        # 질문 유형 분석
        question_type = self.analyze_question_type(question)
        type_examples = self.get_type_specific_examples(question_type)
        type_instructions = self.get_type_specific_instructions(question_type)
        
        prompt = f"""You are an expert Visual Question Answering system with advanced multimodal understanding.

MULTIMODAL FEATURE ANALYSIS:
{feature_info}

QUESTION TYPE: {question_type.upper()}
{type_instructions}

EXAMPLES FOR THIS TYPE:
{type_examples}

CURRENT TASK:
Question: {question}
Choices:
{choices_text}

REASONING PROCESS:
1. Analyze the visual and textual features provided above
2. Consider the specific requirements for {question_type} questions
3. Apply step-by-step logical reasoning
4. Cross-reference with the feature analysis
5. Select the most appropriate answer

Think step by step and provide your reasoning, then conclude with your final answer as a single letter (A, B, C, or D).

ANSWER:"""
        
        return prompt
    
    def analyze_question_type(self, question):
        """질문 유형 분석"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['color', 'size', 'shape', 'appearance']):
            return 'visual_attributes'
        elif any(word in question_lower for word in ['what is', 'what are', 'identify', 'recognize']):
            return 'object_scene_recognition'
        elif any(word in question_lower for word in ['doing', 'activity', 'behavior', 'action']):
            return 'activity_behavior'
        elif any(word in question_lower for word in ['might', 'likely', 'purpose', 'why', 'because']):
            return 'inference_reasoning'
        elif any(word in question_lower for word in ['where', 'when', 'time', 'location']):
            return 'contextual_understanding'
        elif any(word in question_lower for word in ['common', 'typical', 'culture', 'traditional']):
            return 'knowledge_based'
        elif any(word in question_lower for word in ['which', 'best', 'most', 'better']):
            return 'comparison_selection'
        elif any(word in question_lower for word in ['how many', 'count', 'number']):
            return 'quantitative_analysis'
        else:
            return 'general_visual_qa'
    
    def get_type_specific_examples(self, question_type):
        """질문 유형별 예시"""
        examples = {
            'visual_attributes': """
Example 1: Q: What color is the car? A: Focus on color identification from visual features.
Example 2: Q: What shape is the object? A: Analyze geometric properties in the image.""",
            
            'object_scene_recognition': """
Example 1: Q: What is this object? A: Use visual features to identify the main subject.
Example 2: Q: What type of scene is this? A: Consider overall visual composition and context.""",
            
            'activity_behavior': """
Example 1: Q: What is the person doing? A: Analyze human poses and contextual clues.
Example 2: Q: What activity is taking place? A: Look for action indicators and environmental context.""",
            
            'inference_reasoning': """
Example 1: Q: Why might this person be here? A: Consider context clues and logical reasoning.
Example 2: Q: What is the likely purpose? A: Infer from visual evidence and common scenarios.""",
            
            'contextual_understanding': """
Example 1: Q: Where is this taking place? A: Use environmental and contextual visual cues.
Example 2: Q: When might this be happening? A: Consider lighting, clothing, and activity context.""",
            
            'knowledge_based': """
Example 1: Q: What is this traditional item? A: Apply cultural and historical knowledge.
Example 2: Q: What is the common use? A: Consider typical applications and contexts.""",
            
            'comparison_selection': """
Example 1: Q: Which is the best option? A: Compare alternatives based on visual evidence.
Example 2: Q: What is most likely? A: Evaluate probabilities based on visual information.""",
            
            'quantitative_analysis': """
Example 1: Q: How many objects are there? A: Count visible items systematically.
Example 2: Q: What is the approximate number? A: Estimate quantities from visual information.""",
            
            'general_visual_qa': """
Example 1: Q: General question about image? A: Apply comprehensive visual analysis.
Example 2: Q: Mixed-type question? A: Use multi-faceted reasoning approach."""
        }
        
        return examples.get(question_type, examples['general_visual_qa'])
    
    def get_type_specific_instructions(self, question_type):
        """질문 유형별 지시사항"""
        instructions = {
            'visual_attributes': "Focus on visual properties like color, size, shape, texture, and appearance details.",
            'object_scene_recognition': "Identify objects, scenes, or environments using visual classification knowledge.",
            'activity_behavior': "Analyze human actions, behaviors, and activities from visual cues and body language.",
            'inference_reasoning': "Use logical reasoning to infer purposes, motivations, and causal relationships.",
            'contextual_understanding': "Consider spatial, temporal, and situational context from environmental clues.",
            'knowledge_based': "Apply external knowledge about culture, traditions, common practices, and typical uses.",
            'comparison_selection': "Compare options systematically and select based on visual evidence and logic.",
            'quantitative_analysis': "Count, measure, or estimate quantities accurately from visual information.",
            'general_visual_qa': "Apply comprehensive multimodal reasoning combining visual analysis with logical inference."
        }
        
        return instructions.get(question_type, instructions['general_visual_qa'])
    
    def extract_final_answer(self, response):
        """응답에서 최종 답변 추출"""
        import re
        
        # 다양한 패턴으로 답변 추출
        patterns = [
            r'(?:ANSWER|Answer|answer)[:\s]*([ABCD])',
            r'(?:FINAL|Final|final)[:\s]*([ABCD])',
            r'(?:CONCLUSION|Conclusion|conclusion)[:\s]*([ABCD])',
            r'\b([ABCD])\b(?:\s*[.:]|\s*$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response.upper())
            if matches:
                return matches[-1]  # 마지막 매치 사용
        
        return 'A'  # 기본값
    
    def self_verify_answer(self, question, choices_text, feature_info, initial_answer, initial_reasoning):
        """자기 검증"""
        
        verification_prompt = f"""VERIFICATION TASK:
You previously answered a VQA question. Please verify if your answer is correct.

ORIGINAL QUESTION: {question}
CHOICES:
{choices_text}

FEATURE ANALYSIS:
{feature_info}

YOUR INITIAL ANSWER: {initial_answer}
YOUR REASONING: {initial_reasoning}

VERIFICATION INSTRUCTIONS:
1. Re-examine the question and available information
2. Check if your reasoning aligns with the feature analysis
3. Consider if there are any overlooked details
4. Verify logical consistency

Should you change your answer? If yes, provide the new answer (A, B, C, or D).
If no, confirm your original answer.

VERIFICATION RESULT:"""
        
        verification_response = self.generate_text(verification_prompt, max_new_tokens=150, temperature=0.0)
        
        # 검증 결과 확인
        if self.should_change_answer(verification_response):
            new_answer = self.extract_final_answer(verification_response)
            return f"VERIFIED ANSWER: {new_answer} (Changed from {initial_answer})\nREASONING: {verification_response}"
        else:
            return f"VERIFIED ANSWER: {initial_answer} (Confirmed)\nREASONING: {initial_reasoning}"
    
    def should_change_answer(self, verification_response):
        """답변 변경 여부 결정"""
        change_indicators = ['yes', 'change', 'incorrect', 'wrong', 'should be', 'actually']
        confirm_indicators = ['no', 'correct', 'confirm', 'maintain', 'keep']
        
        response_lower = verification_response.lower()
        
        change_count = sum(1 for indicator in change_indicators if indicator in response_lower)
        confirm_count = sum(1 for indicator in confirm_indicators if indicator in response_lower)
        
        return change_count > confirm_count
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """PyTorch 모듈 호환성"""
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)


def load_language_model(**kwargs):
    """언어 모델 로딩 함수"""
    return LanguageModel(**kwargs) 