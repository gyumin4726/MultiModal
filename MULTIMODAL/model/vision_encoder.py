import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional





class VisionEncoder(nn.Module):
    """BLIP2 스타일 Q-Former Vision Encoder"""
    
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 output_dim: int = 768,
                 frozen_stages: int = 1,
                 num_query_tokens: int = 32):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.num_query_tokens = num_query_tokens
        
        # ViT 백본
        print(f"Loading Vision Transformer: {model_name}")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        
        native_dim = self.backbone.embed_dim
        
        # 🔥 BLIP2 스타일 Q-Former 구현
        # 1. Learnable Query Tokens (학습 가능한 쿼리들)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, native_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # 2. Cross-Attention Layer (이미지 패치들과 쿼리들 간 상호작용)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=native_dim,
            num_heads=12,  # ViT와 동일
            dropout=0.1,
            batch_first=True
        )
        
        # 3. Query Transformer (쿼리들을 더 정교하게 처리)
        self.query_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=native_dim,
                nhead=12,
                dim_feedforward=native_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2  # 간단하게 2레이어
        )
        
        # 4. Vision-Language Projection (LLM 차원으로 변환)
        self.vision_language_proj = nn.Sequential(
            nn.Linear(native_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # 기본 전처리 설정
        self.preprocess = self._get_preprocessing()
        
        # 일부 레이어 freeze
        if frozen_stages > 0:
            self._freeze_stages(frozen_stages)
        
        print(f"✅ BLIP2-style Q-Former initialized:")
        print(f"   - {num_query_tokens} learnable query tokens")
        print(f"   - Cross-attention with image patches")
        print(f"   - Query transformer processing")
        print(f"   - Vision-Language projection: {native_dim} -> {output_dim}")
    
    def _freeze_stages(self, frozen_stages: int):
        """지정된 수만큼 초기 레이어들을 freeze"""
        # Patch embedding freeze
        if frozen_stages >= 1:
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
            print(f"❄️ Frozen patch embedding")
        
        # Transformer blocks freeze
        if hasattr(self.backbone, 'blocks'):
            num_blocks_to_freeze = min(frozen_stages - 1, len(self.backbone.blocks))
            for i in range(num_blocks_to_freeze):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = False
            
            if num_blocks_to_freeze > 0:
                print(f"❄️ Frozen first {num_blocks_to_freeze} transformer blocks")
    
    def _get_preprocessing(self):
        """기본 전처리 설정"""
        from torchvision import transforms
        
        # 단일 해상도 사용 (복잡한 멀티스케일 제거)
        size = 224 if 'base' in self.model_name else 384
        
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward_qformer(self, x):
        """BLIP2 스타일 Q-Former 처리"""
        # 이미지 로딩 및 전처리
        if isinstance(x, str):
            from PIL import Image
            image = Image.open(x).convert('RGB')
            processed = self.preprocess(image).unsqueeze(0)
        else:
            processed = x
            
        if torch.cuda.is_available():
            processed = processed.cuda()
        
        # 1. ViT로 이미지 패치 추출
        with torch.no_grad():
            image_patches = self.backbone(processed)  # [1, 197, 768] (CLS + 196 patches)
            # CLS 토큰 제거, 패치만 사용
            if len(image_patches.shape) == 3 and image_patches.shape[1] > 1:
                image_patches = image_patches[:, 1:]  # [1, 196, 768]
        
        # 2. Query tokens를 배치 크기에 맞게 확장
        batch_size = image_patches.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)  # [1, 32, 768]
        
        # 3. Cross-Attention: Query가 이미지 패치들을 참조
        attended_queries, attention_weights = self.cross_attention(
            query_tokens,    # Query: 학습된 쿼리들
            image_patches,   # Key: 이미지 패치들  
            image_patches    # Value: 이미지 패치들
        )  # [1, 32, 768]
        
        # 4. Query Transformer로 쿼리들 간 상호작용
        refined_queries = self.query_transformer(attended_queries)  # [1, 32, 768]
        
        # 5. Vision-Language Projection
        vision_language_features = self.vision_language_proj(refined_queries)  # [1, 32, output_dim]
        
        return vision_language_features, attention_weights
    
    def generate_multimodal_tokens(self, x):
        """이미지를 LLM이 이해할 수 있는 토큰들로 변환 (BLIP2 방식)"""
        vision_tokens, attention_weights = self.forward_qformer(x)  # [1, 32, 768]
        
        # 디버깅용 정보
        token_strengths = torch.mean(torch.abs(vision_tokens), dim=-1)  # [1, 32]
        description = f"Vision tokens: {self.num_query_tokens} learned representations"
        
        return vision_tokens, description
    
    def forward(self, x, question_features=None):
        """메인 forward 함수 - Q-Former 사용"""
        vision_tokens, _ = self.generate_multimodal_tokens(x)
        # 호환성을 위해 평균 pooling으로 단일 벡터 반환
        return torch.mean(vision_tokens, dim=1)  # [1, output_dim]


def load_vision_encoder(model_name: str = 'vit_base_patch16_224', **kwargs) -> VisionEncoder:
    """Vision Encoder 로딩 함수"""
    
    print(f"🖼️ Loading Vision Encoder: {model_name}")
    
    encoder = VisionEncoder(
        model_name=model_name,
        **kwargs
    )
    
    # 파라미터 개수 출력
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"📊 Vision Encoder Stats:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    return encoder


 