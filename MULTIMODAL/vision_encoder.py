import torch
import torch.nn as nn
import sys
import os

# VMamba 모듈 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../mmfscil/models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../VMamba'))

try:
    from vmamba_backbone import VMambaBackbone
    from mamba_ffn_neck import MambaNeck
    VMAMBA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VMamba not available - {e}")
    VMAMBA_AVAILABLE = False


class VisionEncoder(nn.Module):
    """VMamba 기반 비전 인코더 모듈"""
    
    def __init__(self, 
                 model_name='vmamba_tiny_s1l8',
                 pretrained_path='./vssm1_tiny_0230s_ckpt_epoch_264.pth',
                 output_dim=768,
                 frozen_stages=1):
        super().__init__()
        
        if not VMAMBA_AVAILABLE:
            raise ImportError("VMamba components not available. Please check imports.")
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # VMamba 백본 설정
        self.backbone = VMambaBackbone(
            model_name=model_name,
            pretrained_path=pretrained_path,
            out_indices=(0, 1, 2, 3),  # 모든 스테이지에서 피처 추출
            frozen_stages=frozen_stages,
            channel_first=True
        )
        
        # MambaNeck 설정 (비전 피처를 최종 representation으로 변환)
        self.neck = MambaNeck(
            version='ss2d',
            in_channels=768,  # vmamba_tiny_s1l8의 마지막 스테이지 채널 수
            out_channels=output_dim,
            feat_size=7,  # 224 / (4*8) = 7
            num_layers=2,
            use_residual_proj=True,
            # 멀티스케일 스킵 연결 설정
            use_multi_scale_skip=True,
            multi_scale_channels=[96, 192, 384]  # vmamba_tiny_s1l8의 첫 3 스테이지 채널 수
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서 (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: 비전 피처 벡터 (B, output_dim)
        """
        # VMamba 백본을 통해 멀티스케일 피처 추출
        features = self.backbone(x)
        
        # MambaNeck을 통해 최종 representation 생성
        # 마지막 스테이지 피처를 메인으로, 나머지를 multi_scale_features로 사용
        main_feature = features[-1]  # 마지막 스테이지 (B, 768, 7, 7)
        multi_scale_features = features[:-1]  # 첫 3개 스테이지
        
        # MambaNeck 처리
        vision_features = self.neck(main_feature, multi_scale_features)
        
        return vision_features


def load_vision_encoder(**kwargs):
    """비전 인코더 로딩 함수"""
    return VisionEncoder(**kwargs) 