import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# FSCIL 프로젝트의 VMamba 백본을 직접 사용
sys.path.append(os.path.join(os.path.dirname(__file__), '../../mmfscil/models'))
from vmamba_backbone import VMambaBackbone as FSCILVMambaBackbone


class VMambaImageEncoder(nn.Module):
    """FSCIL VMamba 백본을 사용하는 이미지 인코더
    
    train_vmamba_fscil.sh에서 사용하는 것과 동일한 VMambaBackbone을 활용하여
    이미지를 입력받아 고차원 특징 벡터를 출력하는 인코더입니다.
    
    Args:
        model_name (str): VMamba 모델 변형 이름 (train_vmamba_fscil.sh와 동일)
        pretrained_path (str): 사전학습된 가중치 경로
        output_dim (int): 출력 특징 벡터 차원
        frozen_stages (int): 고정할 스테이지 수
        out_indices (tuple): 출력 인덱스 (멀티스케일 특징 추출용)
    """
    
    def __init__(
        self,
        model_name: str = 'vmamba_tiny_s1l8',  # train_vmamba_fscil.sh와 동일
        pretrained_path: Optional[str] = './vssm1_tiny_0230s_ckpt_epoch_264.pth',
        output_dim: int = 768,
        frozen_stages: int = 1,
        out_indices: tuple = (3,),  # 마지막 스테이지만 사용
        channel_first: bool = True,
        image_size: int = 224
    ):
        super().__init__()
        
        self.image_size = image_size
        self.output_dim = output_dim
        self.out_indices = out_indices
        
        # FSCIL VMamba 백본 초기화 (train_vmamba_fscil.sh와 동일한 설정)
        self.backbone = FSCILVMambaBackbone(
            model_name=model_name,
            pretrained_path=pretrained_path,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            channel_first=channel_first
        )
        
        # 백본의 출력 채널 수 확인
        if len(out_indices) == 1:
            backbone_out_channels = self.backbone.out_channels[out_indices[0]]
        else:
            # 멀티스케일인 경우 마지막 스테이지 채널 수 사용
            backbone_out_channels = self.backbone.out_channels[-1]
        
        print(f"VMamba backbone output channels: {backbone_out_channels}")
        
        # 특징 맵을 벡터로 변환하는 어댑터
        # train_vmamba_fscil.sh 설정과 유사한 구조 사용
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # 공간 차원을 7x7로 고정
            nn.Flatten(),  # (B, C, 7, 7) -> (B, C*49)
            nn.Linear(backbone_out_channels * 49, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 위치 인코딩 (선택적) - VMamba의 공간적 이해를 향상
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, 49, output_dim) * 0.02
            )
        
        self._init_weights()
        
        # 모델 정보 출력
        self._print_model_info()
        
    def _print_model_info(self):
        """모델 정보 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"VMamba Image Encoder Info:")
        print(f"  Model: {self.backbone.model_name}")
        print(f"  Pretrained: {self.backbone.pretrained_path}")
        print(f"  Output indices: {self.out_indices}")
        print(f"  Frozen stages: {self.backbone.frozen_stages}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.feature_adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """순전파
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            
        Returns:
            torch.Tensor: 이미지 특징 벡터 (B, output_dim)
        """
        # 이미지 크기 확인 및 리사이즈
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(
                images, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # FSCIL VMamba 백본을 통한 특징 추출
        features = self.backbone(images)
        
        # 단일 출력인 경우 첫 번째 요소 사용
        if len(self.out_indices) == 1:
            feature_map = features[0]  # (B, C, H, W)
        else:
            # 멀티스케일인 경우 마지막 특징 맵 사용
            feature_map = features[-1]  # (B, C, H, W)
        
        # 특징을 벡터로 변환
        image_features = self.feature_adapter(feature_map)  # (B, output_dim)
        
        return image_features
    
    def forward_multiscale(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """멀티스케일 순전파 (모든 스케일 특징 반환)
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            
        Returns:
            Tuple[torch.Tensor, ...]: 각 스케일의 특징 벡터들
        """
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(
                images, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 멀티스케일 설정으로 백본 실행
        temp_out_indices = self.backbone.out_indices
        self.backbone.out_indices = (0, 1, 2, 3)  # 모든 스케일
        
        features = self.backbone(images)  # tuple of feature maps
        
        # 복원
        self.backbone.out_indices = temp_out_indices
        
        # 각 스케일별로 어댑터 적용
        scale_features = []
        for feature_map in features:
            adapted_feature = self.feature_adapter(feature_map)
            scale_features.append(adapted_feature)
        
        return tuple(scale_features)
    
    def get_feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        """특징 맵 반환 (어텐션 시각화 등에 사용)
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            
        Returns:
            torch.Tensor: 특징 맵 (B, C, H, W)
        """
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(
                images, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        features = self.backbone(images)
        
        if len(self.out_indices) == 1:
            return features[0]
        else:
            return features[-1]  # 마지막 스케일 반환
    
    def get_all_feature_maps(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """모든 스케일의 특징 맵 반환
        
        Args:
            images (torch.Tensor): 입력 이미지 (B, 3, H, W)
            
        Returns:
            Tuple[torch.Tensor, ...]: 모든 스케일의 특징 맵들
        """
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(
                images, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 임시로 모든 스케일 출력하도록 설정
        temp_out_indices = self.backbone.out_indices
        self.backbone.out_indices = (0, 1, 2, 3)
        
        features = self.backbone(images)
        
        # 복원
        self.backbone.out_indices = temp_out_indices
        
        return features


def test_vmamba_encoder():
    """VMamba 이미지 인코더 테스트"""
    print("Testing VMamba Image Encoder with FSCIL backbone...")
    
    # 모델 초기화 (train_vmamba_fscil.sh와 동일한 설정)
    encoder = VMambaImageEncoder(
        model_name='vmamba_tiny_s1l8',
        pretrained_path=None,  # 테스트용으로 None
        output_dim=768,
        frozen_stages=1,
        out_indices=(3,)  # 마지막 스테이지만
    )
    
    # 더미 입력
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    # 순전파 테스트
    print("Testing forward pass...")
    with torch.no_grad():
        features = encoder(dummy_images)
        feature_maps = encoder.get_feature_maps(dummy_images)
        all_feature_maps = encoder.get_all_feature_maps(dummy_images)
        multiscale_features = encoder.forward_multiscale(dummy_images)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Feature maps shape: {feature_maps.shape}")
    print(f"All feature maps: {[f.shape for f in all_feature_maps]}")
    print(f"Multiscale features: {[f.shape for f in multiscale_features]}")
    
    print("VMamba Image Encoder test passed!")


if __name__ == "__main__":
    test_vmamba_encoder() 