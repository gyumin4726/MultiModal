import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional


class MultiLayerSkipConnection(nn.Module):
    """MASC-V: Multi-layer Attention Skip Connection for Vision Transformer"""
    
    def __init__(self, feature_dims: List[int], output_dim: int = 1024):
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Cross-attention for multi-layer fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=16,  # ViT-Large에 적합
            dropout=0.1,
            batch_first=True
        )
        
        # Adaptive weights for each layer
        self.layer_weights = nn.Parameter(torch.ones(len(feature_dims)))
        
        # Projection layers for each intermediate feature
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),  # final + skip
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        print(f"MASC-V initialized: {len(feature_dims)} layers -> {output_dim}D")
    
    def forward(self, final_features: torch.Tensor, intermediate_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            final_features: (B, output_dim) - 최종 레이어 피처
            intermediate_features: List of (B, feature_dim) - 중간 레이어 피처들
        
        Returns:
            enhanced_features: (B, output_dim) - Skip connection이 적용된 피처
        """
        batch_size = final_features.size(0)
        
        # 1. 중간 피처들을 output_dim으로 projection
        projected_features = []
        for i, (feat, proj_layer) in enumerate(zip(intermediate_features, self.projections)):
            projected = proj_layer(feat)  # (B, output_dim)
            projected_features.append(projected)
        
        # 2. 가중치 적용 및 결합
        weights = F.softmax(self.layer_weights, dim=0)
        weighted_features = []
        for i, feat in enumerate(projected_features):
            weighted = weights[i] * feat
            weighted_features.append(weighted)
        
        # 3. 중간 피처들을 sequence로 결합
        intermediate_seq = torch.stack(weighted_features, dim=1)  # (B, num_layers, output_dim)
        
        # 4. Cross-attention: final을 query로, intermediate를 key/value로
        final_query = final_features.unsqueeze(1)  # (B, 1, output_dim)
        
        attended_features, attention_weights = self.cross_attention(
            final_query, intermediate_seq, intermediate_seq
        )  # (B, 1, output_dim)
        
        attended_features = attended_features.squeeze(1)  # (B, output_dim)
        
        # 5. Skip connection with fusion
        skip_input = torch.cat([final_features, attended_features], dim=-1)  # (B, output_dim*2)
        enhanced_features = self.fusion_layer(skip_input)  # (B, output_dim)
        
        return enhanced_features


class ViTBackbone(nn.Module):
    """Vision Transformer Backbone with MASC-V support"""
    
    def __init__(self, 
                 model_name: str = 'vit_large_patch16_224',
                 pretrained: bool = True,
                 output_dim: int = 1024,
                 frozen_stages: int = 1,
                 use_skip_connection: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.use_skip_connection = use_skip_connection
        
        # ViT 모델 로딩
        print(f"Loading Vision Transformer: {model_name}")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Feature extraction only
            global_pool='',  # No global pooling - we'll handle it
        )
        
        # 모델 정보 확인
        self.native_dim = self.backbone.embed_dim
        self.num_layers = len(self.backbone.blocks)
        
        print(f"✅ ViT loaded: {self.native_dim}D, {self.num_layers} layers")
        
        # Frozen stages 적용
        if frozen_stages > 0:
            self._freeze_stages(frozen_stages)
        
        # Skip connection을 위한 중간 레이어 인덱스 설정
        if self.use_skip_connection:
            if 'large' in model_name.lower():
                # ViT-Large: 24 layers -> [6, 12, 18, 23] 사용
                self.intermediate_layer_indices = [6, 12, 18, 23]
            elif 'base' in model_name.lower():
                # ViT-Base: 12 layers -> [3, 6, 9, 11] 사용
                self.intermediate_layer_indices = [3, 6, 9, 11]
            elif 'huge' in model_name.lower():
                # ViT-Huge: 32 layers -> [8, 16, 24, 31] 사용
                self.intermediate_layer_indices = [8, 16, 24, 31]
            else:
                # 기본값: 균등 분할
                step = max(1, self.num_layers // 4)
                self.intermediate_layer_indices = [i*step for i in range(1, 4)] + [self.num_layers-1]
            
            # MASC-V 모듈 초기화
            feature_dims = [self.native_dim] * len(self.intermediate_layer_indices)
            self.skip_connection = MultiLayerSkipConnection(feature_dims, output_dim)
            
            print(f"🔗 MASC-V enabled: layers {self.intermediate_layer_indices}")
        
        # Output projection
        if self.native_dim != output_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(self.native_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
        else:
            self.output_projection = nn.Identity()
    
    def _freeze_stages(self, frozen_stages: int):
        """지정된 수만큼 초기 레이어들을 freeze"""
        # Patch embedding freeze
        if frozen_stages >= 1:
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
            print(f"❄️ Frozen patch embedding")
        
        # Transformer blocks freeze
        num_blocks_to_freeze = min(frozen_stages - 1, len(self.backbone.blocks))
        for i in range(num_blocks_to_freeze):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = False
        
        if num_blocks_to_freeze > 0:
            print(f"❄️ Frozen first {num_blocks_to_freeze} transformer blocks")
    
    def forward_features_with_intermediates(self, x):
        """중간 레이어 피처들과 함께 forward"""
        # Patch embedding
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)
        
        intermediate_features = []
        
        # Transformer blocks with intermediate extraction
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            
            # 중간 레이어 피처 저장
            if self.use_skip_connection and i in self.intermediate_layer_indices:
                # Global average pooling for intermediate features
                intermediate_feat = x.mean(dim=1)  # (B, embed_dim)
                intermediate_features.append(intermediate_feat)
        
        # Final processing
        x = self.backbone.norm(x)
        
        # Global average pooling for final features
        final_features = x.mean(dim=1)  # (B, embed_dim)
        
        return final_features, intermediate_features
    
    def forward(self, x):
        """Forward pass with optional skip connections"""
        if self.use_skip_connection:
            # MASC-V 활용
            final_features, intermediate_features = self.forward_features_with_intermediates(x)
            
            # Skip connection 적용
            enhanced_features = self.skip_connection(final_features, intermediate_features)
            
            # Output projection
            output = self.output_projection(enhanced_features)
        else:
            # 기본 forward
            features = self.backbone.forward_features(x)
            if len(features.shape) == 3:  # (B, N, D)
                features = features.mean(dim=1)  # Global average pooling
            
            output = self.output_projection(features)
        
        return output


class VisionEncoder(nn.Module):
    """통합 Vision Encoder with ViT"""
    
    def __init__(self, 
                 model_name: str = 'vit_large_patch16_224',
                 pretrained: bool = True,
                 output_dim: int = 1024,
                 frozen_stages: int = 1,
                 use_skip_connection: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # ViT 백본 사용
        self.backbone = ViTBackbone(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            frozen_stages=frozen_stages,
            use_skip_connection=use_skip_connection
        )
        
        # Image preprocessing
        self.preprocess = self._get_preprocessing()
        
        print(f"✅ VisionEncoder initialized: {model_name} -> {output_dim}D")
    
    def _get_preprocessing(self):
        """이미지 전처리 파이프라인"""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x):
        """
        Args:
            x: 이미지 경로(str) 또는 텐서
        Returns:
            features: (1, output_dim) 피처
        """
        if isinstance(x, str):
            # 이미지 경로인 경우
            from PIL import Image
            image = Image.open(x).convert('RGB')
            x = self.preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)
        
        if torch.cuda.is_available():
            x = x.cuda()
        
        features = self.backbone(x)
        return features


def load_vision_encoder(model_name: str = 'vit_large_patch16_224', **kwargs) -> VisionEncoder:
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


if __name__ == "__main__":
    print("🚀 Vision Transformer with MASC-V Test")
    print("="*50)
    
    # ViT-Large with MASC-V 테스트
    encoder = load_vision_encoder(
        model_name='vit_large_patch16_224',
        pretrained=True,
        output_dim=1024,
        frozen_stages=1,
        use_skip_connection=True
    )
    
    # 더미 이미지로 테스트
    dummy_image = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        dummy_image = dummy_image.cuda()
    
    with torch.no_grad():
        features = encoder.backbone(dummy_image)
    
    print(f"✅ Test passed: Input{dummy_image.shape} -> Output{features.shape}")
    print(f"🔗 MASC-V Skip Connection: {'✅ Enabled' if encoder.backbone.use_skip_connection else '❌ Disabled'}")
    
    if encoder.backbone.use_skip_connection:
        print(f"📍 Intermediate layers: {encoder.backbone.intermediate_layer_indices}")
        print(f"🎯 Enhanced features with multi-layer attention fusion") 