import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np

from .vmamba_backbone import VMambaImageEncoder
from .text_encoder import BERTTextEncoder


class ZeroShotVQAClassifier(nn.Module):
    """Zero-shot VQA 분류기
    
    사전학습된 VMamba와 BERT를 사용하여 학습 없이 VQA를 수행합니다.
    유사도 기반 매칭을 통해 가장 적절한 답변을 선택합니다.
    
    Args:
        image_encoder_config (dict): 이미지 인코더 설정
        text_encoder_config (dict): 텍스트 인코더 설정
        similarity_method (str): 유사도 계산 방법 ('cosine', 'euclidean', 'dot')
        temperature (float): 소프트맥스 온도 파라미터
    """
    
    def __init__(
        self,
        image_encoder_config: Dict = None,
        text_encoder_config: Dict = None,
        similarity_method: str = 'cosine',
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.similarity_method = similarity_method
        self.temperature = temperature
        
        # 기본 설정
        image_encoder_config = image_encoder_config or {}
        text_encoder_config = text_encoder_config or {}
        
        # 이미지 인코더 (train_vmamba_fscil.sh와 동일한 백본)
        self.image_encoder = VMambaImageEncoder(
            model_name=image_encoder_config.get('model_name', 'vmamba_tiny_s1l8'),
            pretrained_path=image_encoder_config.get('pretrained_path', './vssm1_tiny_0230s_ckpt_epoch_264.pth'),
            output_dim=image_encoder_config.get('output_dim', 768),
            frozen_stages=image_encoder_config.get('frozen_stages', 1),
            out_indices=image_encoder_config.get('out_indices', (3,)),
            channel_first=image_encoder_config.get('channel_first', True),
            image_size=image_encoder_config.get('image_size', 224)
        )
        
        # 텍스트 인코더
        self.text_encoder = BERTTextEncoder(
            model_name=text_encoder_config.get('model_name', 'bert-base-uncased'),
            output_dim=text_encoder_config.get('output_dim', 768),
            max_length=text_encoder_config.get('max_length', 512),
            freeze_bert=text_encoder_config.get('freeze_bert', True)  # Zero-shot이므로 고정
        )
        
        # 모든 파라미터를 고정 (학습하지 않음)
        self._freeze_all_parameters()
        
        # 모델 정보 출력
        self._print_model_info()
    
    def _freeze_all_parameters(self):
        """모든 파라미터 고정 (Zero-shot 모드)"""
        for param in self.parameters():
            param.requires_grad = False
        print("All parameters frozen for zero-shot inference")
    
    def _print_model_info(self):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Zero-shot VQA Classifier Info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Similarity method: {self.similarity_method}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Mode: Zero-shot (no training required)")
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """이미지와 텍스트 특징 간 유사도 계산
        
        Args:
            image_features (torch.Tensor): 이미지 특징 (B, D)
            text_features (torch.Tensor): 텍스트 특징 (B, 4, D) 또는 (B*4, D)
            
        Returns:
            torch.Tensor: 유사도 점수 (B, 4)
        """
        if text_features.dim() == 3:  # (B, 4, D)
            batch_size, num_choices, feat_dim = text_features.shape
            # 이미지 특징을 확장: (B, D) -> (B, 4, D)
            expanded_image_features = image_features.unsqueeze(1).expand(-1, num_choices, -1)
            
            if self.similarity_method == 'cosine':
                # 코사인 유사도
                image_norm = F.normalize(expanded_image_features, p=2, dim=-1)
                text_norm = F.normalize(text_features, p=2, dim=-1)
                similarities = torch.sum(image_norm * text_norm, dim=-1)  # (B, 4)
                
            elif self.similarity_method == 'euclidean':
                # 유클리드 거리 (음수로 변환하여 유사도로 사용)
                distances = torch.norm(expanded_image_features - text_features, p=2, dim=-1)
                similarities = -distances  # (B, 4)
                
            elif self.similarity_method == 'dot':
                # 내적
                similarities = torch.sum(expanded_image_features * text_features, dim=-1)  # (B, 4)
                
            else:
                raise ValueError(f"Unknown similarity method: {self.similarity_method}")
                
        else:  # (B*4, D)
            batch_size = image_features.shape[0]
            # 이미지 특징을 4번 반복: (B, D) -> (B*4, D)
            repeated_image_features = image_features.repeat_interleave(4, dim=0)
            
            if self.similarity_method == 'cosine':
                image_norm = F.normalize(repeated_image_features, p=2, dim=-1)
                text_norm = F.normalize(text_features, p=2, dim=-1)
                similarities = torch.sum(image_norm * text_norm, dim=-1)  # (B*4,)
                similarities = similarities.view(batch_size, 4)  # (B, 4)
                
            elif self.similarity_method == 'euclidean':
                distances = torch.norm(repeated_image_features - text_features, p=2, dim=-1)
                similarities = -distances.view(batch_size, 4)  # (B, 4)
                
            elif self.similarity_method == 'dot':
                similarities = torch.sum(repeated_image_features * text_features, dim=-1)
                similarities = similarities.view(batch_size, 4)  # (B, 4)
                
            else:
                raise ValueError(f"Unknown similarity method: {self.similarity_method}")
        
        return similarities
    
    def forward(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str]
    ) -> torch.Tensor:
        """순전파 (Zero-shot 추론)
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 선택지 리스트
            
        Returns:
            torch.Tensor: 유사도 기반 로짓 (B, 4)
        """
        # 이미지 인코딩
        image_features = self.image_encoder(images)  # (B, 768)
        
        # 텍스트 인코딩 (질문+선택지 결합)
        text_outputs = self.text_encoder(
            questions, choices_a, choices_b, choices_c, choices_d,
            encoding_mode='joint'  # 질문과 각 선택지를 결합
        )
        qa_features = text_outputs['qa_features']  # (B, 4, 768)
        
        # 유사도 계산
        similarities = self.compute_similarity(image_features, qa_features)  # (B, 4)
        
        # 온도 스케일링
        logits = similarities / self.temperature
        
        return logits
    
    def predict(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str],
        return_scores: bool = False
    ) -> List[str]:
        """예측 수행
        
        Args:
            images (torch.Tensor): 입력 이미지
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 선택지 리스트
            return_scores (bool): 점수도 함께 반환할지 여부
            
        Returns:
            List[str]: 예측된 답변 ('A', 'B', 'C', 'D')
            또는 Tuple[List[str], torch.Tensor]: (답변, 점수)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                images, questions, choices_a, choices_b, choices_c, choices_d
            )
            
            # 소프트맥스 확률
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)  # (B,)
            
            # 숫자를 문자로 변환
            answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            answers = [answer_map[pred.item()] for pred in predictions]
            
            if return_scores:
                return answers, probabilities
            else:
                return answers
    
    def predict_with_confidence(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str]
    ) -> List[Dict]:
        """신뢰도와 함께 예측 수행
        
        Returns:
            List[Dict]: 각 샘플의 예측 결과
                - answer: 예측 답변
                - confidence: 신뢰도 (최고 확률)
                - probabilities: 모든 선택지 확률
        """
        answers, probabilities = self.predict(
            images, questions, choices_a, choices_b, choices_c, choices_d,
            return_scores=True
        )
        
        results = []
        for i, answer in enumerate(answers):
            probs = probabilities[i].cpu().numpy()
            results.append({
                'answer': answer,
                'confidence': float(probs.max()),
                'probabilities': {
                    'A': float(probs[0]),
                    'B': float(probs[1]),
                    'C': float(probs[2]),
                    'D': float(probs[3])
                }
            })
        
        return results


def test_zero_shot_classifier():
    """Zero-shot VQA 분류기 테스트"""
    print("Testing Zero-shot VQA Classifier...")
    
    # 모델 설정 (train_vmamba_fscil.sh와 동일)
    image_encoder_config = {
        'model_name': 'vmamba_tiny_s1l8',
        'pretrained_path': None,  # 테스트용으로 None
        'output_dim': 768,
        'frozen_stages': 1,
        'out_indices': (3,),
        'channel_first': True,
        'image_size': 224
    }
    
    text_encoder_config = {
        'model_name': 'bert-base-uncased',
        'output_dim': 768,
        'freeze_bert': True
    }
    
    # 모델 초기화
    model = ZeroShotVQAClassifier(
        image_encoder_config=image_encoder_config,
        text_encoder_config=text_encoder_config,
        similarity_method='cosine',
        temperature=1.0
    )
    
    # 더미 데이터
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    questions = [
        "What color is the bird in the image?",
        "How many people are in the photo?"
    ]
    choices_a = ["Red", "One"]
    choices_b = ["Blue", "Two"]
    choices_c = ["Green", "Three"]
    choices_d = ["Yellow", "Four"]
    
    # 추론 테스트
    print("\nTesting zero-shot inference...")
    model.eval()
    with torch.no_grad():
        # 기본 예측
        predictions = model.predict(
            images, questions, choices_a, choices_b, choices_c, choices_d
        )
        
        # 신뢰도와 함께 예측
        detailed_results = model.predict_with_confidence(
            images, questions, choices_a, choices_b, choices_c, choices_d
        )
        
        print(f"Predictions: {predictions}")
        print(f"Detailed results:")
        for i, result in enumerate(detailed_results):
            print(f"  Sample {i}: {result}")
    
    print("\nZero-shot VQA Classifier test passed!")


if __name__ == "__main__":
    test_zero_shot_classifier() 