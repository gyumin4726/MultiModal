import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np


class TextEncoder(nn.Module):
    """transformers 기반 텍스트 인코더 모듈 (PyTorch 1.12.1 호환)"""
    
    def __init__(self, 
                 model_name='sentence-transformers/all-MiniLM-L6-v2',
                 output_dim=768,
                 device=None,
                 max_length=512):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # transformers 라이브러리로 토크나이저와 모델 로딩
        print(f"Loading transformers model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 모델의 원래 임베딩 차원 확인
            self.native_dim = self.model.config.hidden_size
            print(f"Native embedding dimension: {self.native_dim}")
            
        except Exception as e:
            print(f"Failed to load {model_name}, falling back to bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.model.to(self.device)
            self.model.eval()
            self.native_dim = 768
        
        # 출력 차원이 다르면 projection layer 추가
        if self.native_dim != output_dim:
            self.projection = nn.Linear(self.native_dim, output_dim)
            self.projection.to(self.device)
            print(f"Added projection layer: {self.native_dim} -> {output_dim}")
        else:
            self.projection = None
            print(f"No projection needed (native dim = output dim = {output_dim})")
    
    def mean_pooling(self, model_output, attention_mask):
        """평균 풀링을 통한 문장 임베딩 생성"""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def encode_text(self, texts):
        """
        Args:
            texts (list or str): 텍스트 문자열 또는 리스트
            
        Returns:
            torch.Tensor: 텍스트 피처 벡터 (B, output_dim)
        """
        # 단일 문자열이면 리스트로 변환
        if isinstance(texts, str):
            texts = [texts]
        
        # 토크나이징
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # GPU로 이동
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # 모델 추론
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
            # 평균 풀링으로 문장 임베딩 생성
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # L2 정규화
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Projection이 필요하면 적용
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        return embeddings
        
    def forward(self, texts):
        return self.encode_text(texts)


class MultilingualTextEncoder(TextEncoder):
    """다국어 지원 텍스트 인코더"""
    
    def __init__(self, output_dim=768, device=None):
        # 다국어 사전 학습 모델 사용
        super().__init__(
            model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            output_dim=output_dim,
            device=device
        )


class HighPerformanceTextEncoder(TextEncoder):
    """고성능 텍스트 인코더"""
    
    def __init__(self, output_dim=768, device=None):
        # 성능 우수한 사전 학습 모델 사용 (fallback 포함)
        try:
            super().__init__(
                model_name='sentence-transformers/all-mpnet-base-v2',
                output_dim=output_dim,
                device=device
            )
        except:
            # fallback to BERT
            super().__init__(
                model_name='bert-base-uncased',
                output_dim=output_dim,
                device=device
            )


def load_text_encoder(model_type='default', **kwargs):
    """텍스트 인코더 로딩 함수
    
    Args:
        model_type (str): 'default', 'multilingual', 'high_performance'
        **kwargs: 추가 매개변수
    """
    if model_type == 'multilingual':
        return MultilingualTextEncoder(**kwargs)
    elif model_type == 'high_performance':
        return HighPerformanceTextEncoder(**kwargs)
    else:  # default
        return TextEncoder(**kwargs)


# 사용 가능한 사전 학습 모델들 (PyTorch 1.12.1 호환)
AVAILABLE_MODELS = {
    'sentence-transformers/all-MiniLM-L6-v2': {
        'description': '가볍고 빠른 모델 (384차원)',
        'embedding_dim': 384,
        'multilingual': False
    },
    'bert-base-uncased': {
        'description': 'BERT 기본 모델 (768차원)', 
        'embedding_dim': 768,
        'multilingual': False
    },
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': {
        'description': '다국어 지원 모델 (384차원)',
        'embedding_dim': 384,
        'multilingual': True
    },
    'distilbert-base-uncased': {
        'description': 'DistilBERT 기반 (768차원)',
        'embedding_dim': 768,
        'multilingual': False
    }
}

def list_available_models():
    """사용 가능한 사전 학습 모델 목록 출력"""
    print("📋 Available pre-trained text encoder models (PyTorch 1.12.1 compatible):")
    print("-" * 70)
    for model_name, info in AVAILABLE_MODELS.items():
        print(f"🔹 {model_name}")
        print(f"   - {info['description']}")
        print(f"   - Embedding dimension: {info['embedding_dim']}")
        print(f"   - Multilingual: {'Yes' if info['multilingual'] else 'No'}")
        print()


if __name__ == "__main__":
    # 사용 예시
    list_available_models()
    
    # 기본 텍스트 인코더 테스트
    encoder = load_text_encoder()
    
    test_texts = [
        "What is shown in this image?",
        "Describe the main objects in the picture.",
        "Can you identify the key features?"
    ]
    
    embeddings = encoder(test_texts)
    print(f"Text embeddings shape: {embeddings.shape}")
    print(f"Embeddings norm: {embeddings.norm(dim=1)}") 