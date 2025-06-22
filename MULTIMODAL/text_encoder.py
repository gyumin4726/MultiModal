import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np


class TextEncoder(nn.Module):
    """sentence-transformers 기반 텍스트 인코더 모듈"""
    
    def __init__(self, 
                 model_name='all-MiniLM-L6-v2',
                 output_dim=768,
                 device=None):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 사전 학습된 sentence-transformers 모델 로딩
        print(f"Loading sentence-transformers model: {model_name}")
        self.sentence_model = SentenceTransformer(model_name, device=self.device)
        
        # 모델의 원래 임베딩 차원 확인
        self.native_dim = self.sentence_model.get_sentence_embedding_dimension()
        print(f"Native embedding dimension: {self.native_dim}")
        
        # 출력 차원이 다르면 projection layer 추가
        if self.native_dim != output_dim:
            self.projection = nn.Linear(self.native_dim, output_dim)
            print(f"Added projection layer: {self.native_dim} -> {output_dim}")
        else:
            self.projection = None
            print(f"No projection needed (native dim = output dim = {output_dim})")
        
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
        
        # sentence-transformers로 임베딩 생성 (이미 사전 학습된 모델)
        with torch.no_grad():
            embeddings = self.sentence_model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True  # L2 정규화 적용
            )
        
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
            model_name='paraphrase-multilingual-MiniLM-L12-v2',
            output_dim=output_dim,
            device=device
        )


class HighPerformanceTextEncoder(TextEncoder):
    """고성능 텍스트 인코더"""
    
    def __init__(self, output_dim=768, device=None):
        # 성능 우수한 사전 학습 모델 사용
        super().__init__(
            model_name='all-mpnet-base-v2',
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


# 사용 가능한 사전 학습 모델들
AVAILABLE_MODELS = {
    'all-MiniLM-L6-v2': {
        'description': '가볍고 빠른 모델 (384차원)',
        'embedding_dim': 384,
        'multilingual': False
    },
    'all-mpnet-base-v2': {
        'description': '고성능 모델 (768차원)', 
        'embedding_dim': 768,
        'multilingual': False
    },
    'paraphrase-multilingual-MiniLM-L12-v2': {
        'description': '다국어 지원 모델 (384차원)',
        'embedding_dim': 384,
        'multilingual': True
    },
    'all-distilroberta-v1': {
        'description': 'DistilRoBERTa 기반 (768차원)',
        'embedding_dim': 768,
        'multilingual': False
    }
}

def list_available_models():
    """사용 가능한 사전 학습 모델 목록 출력"""
    print("📋 Available pre-trained text encoder models:")
    print("-" * 60)
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