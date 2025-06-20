import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class CrossModalFusion(nn.Module):
    """크로스 모달 퓨전 모듈
    
    이미지와 텍스트 특징을 크로스 어텐션을 통해 융합합니다.
    VMamba의 이미지 특징과 BERT의 텍스트 특징을 효과적으로 결합합니다.
    
    Args:
        image_dim (int): 이미지 특징 차원
        text_dim (int): 텍스트 특징 차원
        hidden_dim (int): 숨겨진 차원
        num_heads (int): 어텐션 헤드 수
        dropout (float): 드롭아웃 비율
        fusion_method (str): 퓨전 방법 ('cross_attention', 'bilinear', 'concat')
    """
    
    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_method: str = 'cross_attention'
    ):
        super().__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fusion_method = fusion_method
        
        # 차원 통일을 위한 투영 레이어
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        if fusion_method == 'cross_attention':
            self._build_cross_attention_layers()
        elif fusion_method == 'bilinear':
            self._build_bilinear_layers()
        elif fusion_method == 'concat':
            self._build_concat_layers()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _build_cross_attention_layers(self):
        """크로스 어텐션 레이어 구축"""
        # 이미지 -> 텍스트 어텐션
        self.img_to_text_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 텍스트 -> 이미지 어텐션
        self.text_to_img_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 셀프 어텐션 (융합된 특징 정제)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 피드포워드 네트워크
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
    
    def _build_bilinear_layers(self):
        """바이리니어 퓨전 레이어 구축"""
        self.bilinear = nn.Bilinear(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
    
    def _build_concat_layers(self):
        """연결 기반 퓨전 레이어 구축"""
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def cross_attention_fusion(
        self, 
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """크로스 어텐션 기반 퓨전
        
        Args:
            image_features (torch.Tensor): 이미지 특징 (B, image_dim)
            text_features (torch.Tensor): 텍스트 특징 (B, N, text_dim) 또는 (B, text_dim)
            
        Returns:
            torch.Tensor: 융합된 특징 (B, hidden_dim)
        """
        batch_size = image_features.shape[0]
        
        # 차원 투영
        img_proj = self.image_proj(image_features)  # (B, hidden_dim)
        
        # 텍스트 특징이 2D인 경우 (단일 텍스트)
        if text_features.dim() == 2:
            text_proj = self.text_proj(text_features)  # (B, hidden_dim)
            # 시퀀스 차원 추가
            img_proj = img_proj.unsqueeze(1)  # (B, 1, hidden_dim)
            text_proj = text_proj.unsqueeze(1)  # (B, 1, hidden_dim)
        else:
            # 텍스트 특징이 3D인 경우 (여러 텍스트, 예: 질문 + 선택지들)
            text_proj = self.text_proj(text_features)  # (B, N, hidden_dim)
            img_proj = img_proj.unsqueeze(1).expand(-1, text_proj.shape[1], -1)  # (B, N, hidden_dim)
        
        # 크로스 어텐션: 이미지 -> 텍스트
        img_attended, _ = self.img_to_text_attention(
            query=img_proj,
            key=text_proj,
            value=text_proj
        )
        img_attended = self.norm1(img_proj + self.dropout(img_attended))
        
        # 크로스 어텐션: 텍스트 -> 이미지
        text_attended, _ = self.text_to_img_attention(
            query=text_proj,
            key=img_proj,
            value=img_proj
        )
        text_attended = self.norm2(text_proj + self.dropout(text_attended))
        
        # 융합된 특징 결합
        if text_features.dim() == 2:
            # 단일 텍스트인 경우
            fused = (img_attended + text_attended) / 2  # (B, 1, hidden_dim)
            fused = fused.squeeze(1)  # (B, hidden_dim)
        else:
            # 여러 텍스트인 경우 (선택지들)
            fused = (img_attended + text_attended) / 2  # (B, N, hidden_dim)
            # 평균 풀링 또는 최대 풀링
            fused = fused.mean(dim=1)  # (B, hidden_dim)
        
        # 셀프 어텐션으로 정제
        fused_input = fused.unsqueeze(1)  # (B, 1, hidden_dim)
        fused_refined, _ = self.self_attention(
            query=fused_input,
            key=fused_input,
            value=fused_input
        )
        fused_refined = self.norm3(fused_input + self.dropout(fused_refined))
        fused_refined = fused_refined.squeeze(1)  # (B, hidden_dim)
        
        # 피드포워드 네트워크
        fused_final = fused_refined + self.dropout(self.ffn(fused_refined))
        fused_final = self.layer_norm(fused_final)
        
        return fused_final
    
    def bilinear_fusion(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """바이리니어 퓨전
        
        Args:
            image_features (torch.Tensor): 이미지 특징 (B, image_dim)
            text_features (torch.Tensor): 텍스트 특징 (B, text_dim) 또는 (B, N, text_dim)
            
        Returns:
            torch.Tensor: 융합된 특징 (B, hidden_dim)
        """
        # 차원 투영
        img_proj = self.image_proj(image_features)  # (B, hidden_dim)
        
        if text_features.dim() == 3:
            # 여러 텍스트인 경우 평균 풀링
            text_features = text_features.mean(dim=1)  # (B, text_dim)
        
        text_proj = self.text_proj(text_features)  # (B, hidden_dim)
        
        # 바이리니어 퓨전
        fused = self.bilinear(img_proj, text_proj)  # (B, hidden_dim)
        fused = self.fusion_proj(fused)
        fused = self.layer_norm(fused)
        
        return fused
    
    def concat_fusion(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """연결 기반 퓨전
        
        Args:
            image_features (torch.Tensor): 이미지 특징 (B, image_dim)
            text_features (torch.Tensor): 텍스트 특징 (B, text_dim) 또는 (B, N, text_dim)
            
        Returns:
            torch.Tensor: 융합된 특징 (B, hidden_dim)
        """
        # 차원 투영
        img_proj = self.image_proj(image_features)  # (B, hidden_dim)
        
        if text_features.dim() == 3:
            # 여러 텍스트인 경우 평균 풀링
            text_features = text_features.mean(dim=1)  # (B, text_dim)
        
        text_proj = self.text_proj(text_features)  # (B, hidden_dim)
        
        # 연결 후 투영
        concatenated = torch.cat([img_proj, text_proj], dim=1)  # (B, hidden_dim * 2)
        fused = self.fusion_proj(concatenated)  # (B, hidden_dim)
        fused = self.layer_norm(fused)
        
        return fused
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """순전파
        
        Args:
            image_features (torch.Tensor): 이미지 특징
            text_features (torch.Tensor): 텍스트 특징
            
        Returns:
            torch.Tensor: 융합된 특징
        """
        if self.fusion_method == 'cross_attention':
            return self.cross_attention_fusion(image_features, text_features)
        elif self.fusion_method == 'bilinear':
            return self.bilinear_fusion(image_features, text_features)
        elif self.fusion_method == 'concat':
            return self.concat_fusion(image_features, text_features)


class AdaptiveFusion(nn.Module):
    """적응적 퓨전 모듈
    
    여러 퓨전 방법을 결합하고 가중치를 학습하여 최적의 융합을 수행합니다.
    """
    
    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 여러 퓨전 방법들
        self.cross_attention_fusion = CrossModalFusion(
            image_dim, text_dim, hidden_dim, num_heads, dropout, 'cross_attention'
        )
        self.bilinear_fusion = CrossModalFusion(
            image_dim, text_dim, hidden_dim, num_heads, dropout, 'bilinear'
        )
        self.concat_fusion = CrossModalFusion(
            image_dim, text_dim, hidden_dim, num_heads, dropout, 'concat'
        )
        
        # 퓨전 방법들의 가중치를 학습
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
        # 최종 투영
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """순전파
        
        Args:
            image_features (torch.Tensor): 이미지 특징
            text_features (torch.Tensor): 텍스트 특징
            
        Returns:
            torch.Tensor: 융합된 특징
        """
        # 각 퓨전 방법 적용
        cross_fused = self.cross_attention_fusion(image_features, text_features)
        bilinear_fused = self.bilinear_fusion(image_features, text_features)
        concat_fused = self.concat_fusion(image_features, text_features)
        
        # 가중치 정규화
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # 가중 합
        fused = (weights[0] * cross_fused + 
                weights[1] * bilinear_fused + 
                weights[2] * concat_fused)
        
        # 최종 투영
        fused = self.final_proj(fused)
        
        return fused


def test_fusion_module():
    """퓨전 모듈 테스트"""
    print("Testing Cross Modal Fusion Module...")
    
    # 모델 초기화
    fusion = CrossModalFusion(
        image_dim=768,
        text_dim=768,
        hidden_dim=768,
        fusion_method='cross_attention'
    )
    
    # 더미 데이터
    batch_size = 2
    image_features = torch.randn(batch_size, 768)
    text_features_2d = torch.randn(batch_size, 768)  # 단일 텍스트
    text_features_3d = torch.randn(batch_size, 4, 768)  # 4개 선택지
    
    # 순전파 테스트
    with torch.no_grad():
        fused_2d = fusion(image_features, text_features_2d)
        fused_3d = fusion(image_features, text_features_3d)
    
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features 2D shape: {text_features_2d.shape}")
    print(f"Text features 3D shape: {text_features_3d.shape}")
    print(f"Fused features 2D shape: {fused_2d.shape}")
    print(f"Fused features 3D shape: {fused_3d.shape}")
    print(f"Model parameters: {sum(p.numel() for p in fusion.parameters()):,}")
    
    # 적응적 퓨전 테스트
    print("\nTesting Adaptive Fusion...")
    adaptive_fusion = AdaptiveFusion()
    
    with torch.no_grad():
        adaptive_fused = adaptive_fusion(image_features, text_features_2d)
    
    print(f"Adaptive fused shape: {adaptive_fused.shape}")
    print(f"Fusion weights: {F.softmax(adaptive_fusion.fusion_weights, dim=0)}")
    
    print("Cross Modal Fusion test passed!")


if __name__ == "__main__":
    test_fusion_module() 