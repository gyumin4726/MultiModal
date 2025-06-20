import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import warnings

# Transformers 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)


class BERTTextEncoder(nn.Module):
    """BERT 기반 텍스트 인코더
    
    질문과 선택지를 인코딩하여 텍스트 특징을 추출합니다.
    2024년 이전 공개된 BERT 모델을 사용하여 대회 규칙을 준수합니다.
    
    Args:
        model_name (str): 사용할 BERT 모델 이름
        output_dim (int): 출력 특징 벡터 차원
        max_length (int): 최대 토큰 길이
        freeze_bert (bool): BERT 레이어 고정 여부
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        output_dim: int = 768,
        max_length: int = 512,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.freeze_bert = freeze_bert
        
        # BERT 토크나이저와 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        # BERT 출력 차원
        self.bert_dim = self.bert_model.config.hidden_size
        
        # BERT 레이어 고정
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        
        # 텍스트 특징 투영 레이어
        self.text_projector = nn.Sequential(
            nn.Linear(self.bert_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 선택지 인코딩을 위한 추가 레이어
        self.choice_projector = nn.Sequential(
            nn.Linear(self.bert_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in [self.text_projector, self.choice_projector]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """텍스트 리스트를 인코딩
        
        Args:
            texts (List[str]): 인코딩할 텍스트 리스트
            
        Returns:
            torch.Tensor: 텍스트 특징 (B, output_dim)
        """
        # 토크나이징
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 모델과 같은 디바이스로 이동
        device = next(self.bert_model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # BERT 인코딩
        with torch.set_grad_enabled(not self.freeze_bert):
            outputs = self.bert_model(**encoded)
            pooled_output = outputs.pooler_output  # (B, bert_dim)
        
        # 특징 투영
        text_features = self.text_projector(pooled_output)  # (B, output_dim)
        
        return text_features
    
    def encode_question_and_choices(
        self, 
        questions: List[str], 
        choices_a: List[str],
        choices_b: List[str], 
        choices_c: List[str],
        choices_d: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """질문과 선택지들을 함께 인코딩
        
        Args:
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 각 선택지 리스트
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 질문 특징 (B, output_dim)
                - 선택지 특징 (B, 4, output_dim)
        """
        # 질문 인코딩
        question_features = self.encode_text(questions)
        
        # 각 선택지 인코딩
        choice_features = []
        for choices in [choices_a, choices_b, choices_c, choices_d]:
            choice_feat = self.encode_choice(choices)
            choice_features.append(choice_feat)
        
        # 선택지 특징을 스택
        choice_features = torch.stack(choice_features, dim=1)  # (B, 4, output_dim)
        
        return question_features, choice_features
    
    def encode_choice(self, choices: List[str]) -> torch.Tensor:
        """선택지 인코딩 (질문 인코딩과 약간 다른 처리)
        
        Args:
            choices (List[str]): 선택지 리스트
            
        Returns:
            torch.Tensor: 선택지 특징 (B, output_dim)
        """
        # 토크나이징
        encoded = self.tokenizer(
            choices,
            padding=True,
            truncation=True,
            max_length=self.max_length // 2,  # 선택지는 보통 더 짧음
            return_tensors='pt'
        )
        
        # 모델과 같은 디바이스로 이동
        device = next(self.bert_model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # BERT 인코딩
        with torch.set_grad_enabled(not self.freeze_bert):
            outputs = self.bert_model(**encoded)
            pooled_output = outputs.pooler_output  # (B, bert_dim)
        
        # 선택지 특징 투영
        choice_features = self.choice_projector(pooled_output)  # (B, output_dim)
        
        return choice_features
    
    def encode_qa_pairs(
        self,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str],
        choices_c: List[str], 
        choices_d: List[str]
    ) -> torch.Tensor:
        """질문-선택지 쌍을 함께 인코딩 (더 나은 성능을 위해)
        
        Args:
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 각 선택지 리스트
            
        Returns:
            torch.Tensor: QA 쌍 특징 (B, 4, output_dim)
        """
        batch_size = len(questions)
        all_qa_pairs = []
        
        # 각 선택지와 질문을 결합
        for i in range(batch_size):
            qa_pairs = []
            for choice in [choices_a[i], choices_b[i], choices_c[i], choices_d[i]]:
                # 질문과 선택지를 [SEP] 토큰으로 결합
                qa_pair = f"{questions[i]} [SEP] {choice}"
                qa_pairs.append(qa_pair)
            all_qa_pairs.extend(qa_pairs)
        
        # 모든 QA 쌍을 한번에 인코딩
        qa_features = self.encode_text(all_qa_pairs)  # (B*4, output_dim)
        
        # 배치별로 재구성
        qa_features = qa_features.view(batch_size, 4, self.output_dim)  # (B, 4, output_dim)
        
        return qa_features
    
    def forward(
        self,
        questions: List[str],
        choices_a: List[str],
        choices_b: List[str], 
        choices_c: List[str],
        choices_d: List[str],
        encoding_mode: str = 'separate'
    ) -> Dict[str, torch.Tensor]:
        """순전파
        
        Args:
            questions (List[str]): 질문 리스트
            choices_a, choices_b, choices_c, choices_d (List[str]): 각 선택지 리스트
            encoding_mode (str): 인코딩 모드 ('separate' 또는 'joint')
            
        Returns:
            Dict[str, torch.Tensor]: 텍스트 특징들
        """
        if encoding_mode == 'separate':
            question_features, choice_features = self.encode_question_and_choices(
                questions, choices_a, choices_b, choices_c, choices_d
            )
            return {
                'question_features': question_features,
                'choice_features': choice_features
            }
        elif encoding_mode == 'joint':
            qa_features = self.encode_qa_pairs(
                questions, choices_a, choices_b, choices_c, choices_d
            )
            return {
                'qa_features': qa_features
            }
        else:
            raise ValueError(f"Unknown encoding mode: {encoding_mode}")


def test_text_encoder():
    """텍스트 인코더 테스트"""
    print("Testing BERT Text Encoder...")
    
    # 모델 초기화
    encoder = BERTTextEncoder(
        model_name='bert-base-uncased',
        output_dim=768
    )
    
    # 더미 데이터
    questions = [
        "What color is the bird in the image?",
        "How many people are in the photo?"
    ]
    choices_a = ["Red", "One"]
    choices_b = ["Blue", "Two"] 
    choices_c = ["Green", "Three"]
    choices_d = ["Yellow", "Four"]
    
    # 순전파 테스트
    with torch.no_grad():
        # Separate 모드
        outputs_sep = encoder(
            questions, choices_a, choices_b, choices_c, choices_d,
            encoding_mode='separate'
        )
        
        # Joint 모드
        outputs_joint = encoder(
            questions, choices_a, choices_b, choices_c, choices_d,
            encoding_mode='joint'
        )
    
    print(f"Questions: {questions}")
    print(f"Separate mode - Question features shape: {outputs_sep['question_features'].shape}")
    print(f"Separate mode - Choice features shape: {outputs_sep['choice_features'].shape}")
    print(f"Joint mode - QA features shape: {outputs_joint['qa_features'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("BERT Text Encoder test passed!")


if __name__ == "__main__":
    test_text_encoder() 