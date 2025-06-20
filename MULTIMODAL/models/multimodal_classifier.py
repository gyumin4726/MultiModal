import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import math

from .vmamba_backbone import VMambaImageEncoder
from .text_encoder import BERTTextEncoder
from .fusion_module import CrossModalFusion, AdaptiveFusion


class MultiModalVQAClassifier(nn.Module):
    """멀티모달 VQA 분류기
    
    VMamba 기반 이미지 인코더와 BERT 기반 텍스트 인코더를 결합하여
    Visual Question Answering 태스크를 수행하는 통합 모델입니다.
    
    Args:
        image_encoder_config (dict): 이미지 인코더 설정
        text_encoder_config (dict): 텍스트 인코더 설정
        fusion_config (dict): 퓨전 모듈 설정
        num_classes (int): 분류할 클래스 수 (4: A, B, C, D)
        dropout (float): 드롭아웃 비율
    """
    
    def __init__(
        self,
        image_encoder_config: Dict = None,
        text_encoder_config: Dict = None,
        fusion_config: Dict = None,
        num_classes: int = 4,
        dropout: float = 0.1,
        use_adaptive_fusion: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_adaptive_fusion = use_adaptive_fusion
        
        # 기본 설정
        image_encoder_config = image_encoder_config or {}
        text_encoder_config = text_encoder_config or {}
        fusion_config = fusion_config or {}
        
        # 이미지 인코더 (train_vmamba_fscil.sh와 동일한 백본 사용)
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
            freeze_bert=text_encoder_config.get('freeze_bert', False)
        )
        
        # 퓨전 모듈
        if use_adaptive_fusion:
            self.fusion_module = AdaptiveFusion(
                image_dim=image_encoder_config.get('output_dim', 768),
                text_dim=text_encoder_config.get('output_dim', 768),
                hidden_dim=fusion_config.get('hidden_dim', 768),
                num_heads=fusion_config.get('num_heads', 8),
                dropout=dropout
            )
        else:
            self.fusion_module = CrossModalFusion(
                image_dim=image_encoder_config.get('output_dim', 768),
                text_dim=text_encoder_config.get('output_dim', 768),
                hidden_dim=fusion_config.get('hidden_dim', 768),
                num_heads=fusion_config.get('num_heads', 8),
                dropout=dropout,
                fusion_method=fusion_config.get('fusion_method', 'cross_attention')
            )
        
        # 분류 헤드
        hidden_dim = fusion_config.get('hidden_dim', 768)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 가중치 초기화
        self._init_weights()
        
        # 모델 정보 출력
        self._print_model_info()
    
    def _init_weights(self):
        """분류기 가중치 초기화"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _print_model_info(self):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"MultiModal VQA Classifier Info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 3B 파라미터 제한 확인
        if total_params >= 3_000_000_000:
            print(f"  WARNING: Model exceeds 3B parameter limit!")
        else:
            print(f"  ✓ Model is within 3B parameter limit")
    
    def forward_separate_encoding(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str]
    ) -> torch.Tensor:
        """질문과 선택지를 별도로 인코딩하는 방식
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 선택지 리스트
            
        Returns:
            torch.Tensor: 분류 로짓 (B, 4)
        """
        # 이미지 인코딩
        image_features = self.image_encoder(images)  # (B, 768)
        
        # 텍스트 인코딩 (질문과 선택지 별도)
        text_outputs = self.text_encoder(
            questions, choices_a, choices_b, choices_c, choices_d,
            encoding_mode='separate'
        )
        question_features = text_outputs['question_features']  # (B, 768)
        choice_features = text_outputs['choice_features']      # (B, 4, 768)
        
        # 각 선택지별로 이미지-질문-선택지 융합
        batch_size = images.shape[0]
        choice_scores = []
        
        for i in range(4):  # A, B, C, D
            # 질문과 해당 선택지 결합
            combined_text = question_features + choice_features[:, i, :]  # (B, 768)
            
            # 이미지와 텍스트 융합
            fused_features = self.fusion_module(image_features, combined_text)  # (B, 768)
            
            # 해당 선택지에 대한 점수 계산
            choice_score = self.classifier(fused_features)  # (B, 4)
            choice_scores.append(choice_score[:, i:i+1])  # (B, 1)
        
        # 모든 선택지 점수 결합
        logits = torch.cat(choice_scores, dim=1)  # (B, 4)
        
        return logits
    
    def forward_joint_encoding(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str]
    ) -> torch.Tensor:
        """질문과 선택지를 함께 인코딩하는 방식
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 선택지 리스트
            
        Returns:
            torch.Tensor: 분류 로짓 (B, 4)
        """
        # 이미지 인코딩
        image_features = self.image_encoder(images)  # (B, 768)
        
        # 텍스트 인코딩 (질문-선택지 쌍으로)
        text_outputs = self.text_encoder(
            questions, choices_a, choices_b, choices_c, choices_d,
            encoding_mode='joint'
        )
        qa_features = text_outputs['qa_features']  # (B, 4, 768)
        
        # 각 QA 쌍별로 이미지와 융합
        batch_size = images.shape[0]
        choice_scores = []
        
        for i in range(4):  # A, B, C, D
            # 이미지와 해당 QA 쌍 융합
            fused_features = self.fusion_module(
                image_features, qa_features[:, i, :]
            )  # (B, 768)
            
            # 해당 선택지에 대한 점수 계산
            choice_score = self.classifier(fused_features)  # (B, 4)
            choice_scores.append(choice_score[:, i:i+1])  # (B, 1)
        
        # 모든 선택지 점수 결합
        logits = torch.cat(choice_scores, dim=1)  # (B, 4)
        
        return logits
    
    def forward_unified_encoding(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str]
    ) -> torch.Tensor:
        """통합된 인코딩 방식 (더 효율적)
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 선택지 리스트
            
        Returns:
            torch.Tensor: 분류 로짓 (B, 4)
        """
        # 이미지 인코딩
        image_features = self.image_encoder(images)  # (B, 768)
        
        # 텍스트 인코딩 (질문-선택지 쌍으로)
        text_outputs = self.text_encoder(
            questions, choices_a, choices_b, choices_c, choices_d,
            encoding_mode='joint'
        )
        qa_features = text_outputs['qa_features']  # (B, 4, 768)
        
        # 이미지 특징을 4개 선택지에 맞게 확장
        expanded_image_features = image_features.unsqueeze(1).expand(-1, 4, -1)  # (B, 4, 768)
        
        # 배치 차원으로 펼치기
        batch_size = images.shape[0]
        flat_image_features = expanded_image_features.view(-1, expanded_image_features.shape[-1])  # (B*4, 768)
        flat_qa_features = qa_features.view(-1, qa_features.shape[-1])  # (B*4, 768)
        
        # 융합
        fused_features = self.fusion_module(flat_image_features, flat_qa_features)  # (B*4, 768)
        
        # 분류
        flat_logits = self.classifier(fused_features)  # (B*4, 4)
        
        # 배치 차원으로 복원하고 대각선 요소만 추출
        logits = flat_logits.view(batch_size, 4, 4)  # (B, 4, 4)
        final_logits = torch.diagonal(logits, dim1=1, dim2=2)  # (B, 4)
        
        return final_logits
    
    def forward(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str],
        encoding_mode: str = 'unified'
    ) -> torch.Tensor:
        """순전파
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 선택지 리스트
            encoding_mode (str): 인코딩 모드 ('separate', 'joint', 'unified')
            
        Returns:
            torch.Tensor: 분류 로짓 (B, 4)
        """
        if encoding_mode == 'separate':
            return self.forward_separate_encoding(
                images, questions, choices_a, choices_b, choices_c, choices_d
            )
        elif encoding_mode == 'joint':
            return self.forward_joint_encoding(
                images, questions, choices_a, choices_b, choices_c, choices_d
            )
        elif encoding_mode == 'unified':
            return self.forward_unified_encoding(
                images, questions, choices_a, choices_b, choices_c, choices_d
            )
        else:
            raise ValueError(f"Unknown encoding mode: {encoding_mode}")
    
    def predict(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str],
        encoding_mode: str = 'unified'
    ) -> List[str]:
        """예측 수행
        
        Args:
            images (torch.Tensor): 입력 이미지
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 선택지 리스트
            encoding_mode (str): 인코딩 모드
            
        Returns:
            List[str]: 예측된 답변 ('A', 'B', 'C', 'D')
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                images, questions, choices_a, choices_b, choices_c, choices_d,
                encoding_mode=encoding_mode
            )
            predictions = torch.argmax(logits, dim=1)  # (B,)
            
            # 숫자를 문자로 변환
            answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            answers = [answer_map[pred.item()] for pred in predictions]
            
        return answers
    
    def get_attention_weights(
        self,
        images: torch.Tensor,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str],
        choices_d: List[str]
    ) -> Dict[str, torch.Tensor]:
        """어텐션 가중치 반환 (시각화용)
        
        Returns:
            Dict[str, torch.Tensor]: 어텐션 가중치들
        """
        # 이 기능은 퓨전 모듈에서 어텐션 가중치를 반환하도록 수정 필요
        # 현재는 간단한 버전으로 구현
        return {}


def test_multimodal_classifier():
    """멀티모달 분류기 테스트 (train_vmamba_fscil.sh와 동일한 백본 사용)"""
    print("Testing MultiModal VQA Classifier with FSCIL VMamba backbone...")
    
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
        'freeze_bert': False
    }
    
    fusion_config = {
        'hidden_dim': 768,
        'num_heads': 8,
        'fusion_method': 'cross_attention'
    }
    
    # 모델 초기화
    model = MultiModalVQAClassifier(
        image_encoder_config=image_encoder_config,
        text_encoder_config=text_encoder_config,
        fusion_config=fusion_config,
        num_classes=4,
        use_adaptive_fusion=False
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
    
    # 순전파 테스트
    model.eval()
    with torch.no_grad():
        # 다양한 인코딩 모드 테스트
        for mode in ['unified', 'joint', 'separate']:
            print(f"\nTesting {mode} encoding mode...")
            logits = model(
                images, questions, choices_a, choices_b, choices_c, choices_d,
                encoding_mode=mode
            )
            predictions = model.predict(
                images, questions, choices_a, choices_b, choices_c, choices_d,
                encoding_mode=mode
            )
            
            print(f"  Logits shape: {logits.shape}")
            print(f"  Predictions: {predictions}")
            print(f"  Probabilities: {F.softmax(logits, dim=1)}")
    
    print("\nMultiModal VQA Classifier test passed!")


if __name__ == "__main__":
    test_multimodal_classifier() 