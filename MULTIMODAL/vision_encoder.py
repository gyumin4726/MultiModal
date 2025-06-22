import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import warnings

# VMamba 관련 임포트 시도
VMAMBA_AVAILABLE = False
try:
    # vmamba.py에서 필요한 클래스들을 직접 임포트
    import sys
    import os
    
    # VMamba 파일이 있는지 확인
    vmamba_path = os.path.join(os.path.dirname(__file__), 'vmamba.py')
    if os.path.exists(vmamba_path):
        sys.path.insert(0, os.path.dirname(__file__))
        from vmamba import vmamba_tiny_s1l8, VSSM
        VMAMBA_AVAILABLE = True
        print("✅ VMamba loaded successfully from vmamba.py")
    else:
        print("❌ vmamba.py not found in current directory")
except ImportError as e:
    print(f"Warning: VMamba import failed - {e}")
except Exception as e:
    print(f"Warning: VMamba loading error - {e}")

# Fallback: ResNet 백본 (항상 사용 가능)
try:
    import torchvision.models as models
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False


class VMambaBackbone(nn.Module):
    """VMamba 백본 (vmamba.py 사용)"""
    
    def __init__(self, model_name='vmamba_tiny_s1l8', pretrained=True, output_dim=768):
        super().__init__()
        
        if not VMAMBA_AVAILABLE:
            raise ImportError("VMamba not available. Please ensure vmamba.py is in the current directory.")
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # VMamba 모델 생성
        print(f"🔧 Creating VMamba model: {model_name}")
        if model_name == 'vmamba_tiny_s1l8':
            print(f"   Using vmamba_tiny_s1l8 with pretrained={pretrained}")
            self.backbone = vmamba_tiny_s1l8(pretrained=pretrained, channel_first=True)
            print(f"   ✅ vmamba_tiny_s1l8 model created successfully")
        else:
            print(f"   Using VSSM class with custom config")
            # 기본적으로 VSSM 클래스 사용
            self.backbone = VSSM(
                depths=[2, 2, 8, 2], 
                dims=96, 
                drop_path_rate=0.2,
                patch_size=4, 
                in_chans=3, 
                num_classes=1000,
                ssm_d_state=1, 
                ssm_ratio=1.0, 
                ssm_dt_rank="auto", 
                ssm_act_layer="silu",
                ssm_conv=3, 
                ssm_conv_bias=False, 
                ssm_drop_rate=0.0,
                ssm_init="v0", 
                forward_type="v05_noz",
                mlp_ratio=4.0, 
                mlp_act_layer="gelu", 
                mlp_drop_rate=0.0, 
                gmlp=False,
                patch_norm=True, 
                norm_layer="ln2d",
                downsample_version="v3", 
                patchembed_version="v2",
                use_checkpoint=False, 
                posembed=False, 
                imgsize=224,
            )
        
        # classifier 제거하고 feature extractor로 사용
        if hasattr(self.backbone, 'classifier'):
            del self.backbone.classifier
        
        # 마지막 레이어의 차원 확인
        self.feature_dim = self._get_feature_dim()
        
        # 출력 차원 조정을 위한 projection layer
        if self.feature_dim != output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = None
            
        print(f"VMamba backbone initialized: {model_name}")
        print(f"Feature dimension: {self.feature_dim} -> {output_dim}")
    
    def load_pretrained_weights(self, pretrained_path):
        """VMamba 전용 가중치 로딩"""
        print(f"📥 Loading VMamba weights from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # 직접 backbone에 로딩 (접두사 없이)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            
            print(f"   ✅ Loaded VMamba weights")
            print(f"   📊 Missing keys: {len(missing_keys)}")
            print(f"   📊 Unexpected keys: {len(unexpected_keys)}")
            
            if len(missing_keys) > 0:
                print(f"   ⚠️ Missing keys (first 5): {missing_keys[:5]}")
            if len(unexpected_keys) > 0:
                print(f"   ⚠️ Unexpected keys (first 5): {unexpected_keys[:5]}")
                
        except Exception as e:
            print(f"   ❌ Failed to load VMamba weights: {e}")
    
    def _get_feature_dim(self):
        """백본의 피처 차원 확인"""
        try:
            # 더미 입력으로 차원 확인
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = self.extract_features(dummy_input)
                return features.shape[1]
        except:
            # 기본값 반환
            return 768
    
    def extract_features(self, x):
        """VMamba에서 피처 추출"""
        # patch embedding
        x = self.backbone.patch_embed(x)
        
        # position embedding (있다면)
        if hasattr(self.backbone, 'pos_embed') and self.backbone.pos_embed is not None:
            pos_embed = self.backbone.pos_embed
            x = x + pos_embed
        
        # 각 레이어 통과
        for layer in self.backbone.layers:
            x = layer(x)
        
        # Global Average Pooling
        if len(x.shape) == 4:  # (B, C, H, W)
            x = torch.mean(x, dim=(2, 3))  # (B, C)
        elif len(x.shape) == 3:  # (B, H*W, C) 또는 (B, C, H*W)
            if x.shape[1] > x.shape[2]:  # (B, H*W, C)
                x = torch.mean(x, dim=1)  # (B, C)
            else:  # (B, C, H*W)
                x = torch.mean(x, dim=2)  # (B, C)
        
        return x
    
    def forward(self, x):
        features = self.extract_features(x)
        
        if self.projection is not None:
            features = self.projection(features)
            
        return features


class ResNetBackbone(nn.Module):
    """ResNet 백본 (Fallback)"""
    
    def __init__(self, model_name='resnet50', pretrained=True, output_dim=768):
        super().__init__()
        
        if not RESNET_AVAILABLE:
            raise ImportError("torchvision not available")
        
        # ResNet 모델 로딩
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        
        # classifier 제거
        self.backbone.fc = nn.Identity()
        
        # 출력 차원 조정
        if feature_dim != output_dim:
            self.projection = nn.Linear(feature_dim, output_dim)
        else:
            self.projection = None
            
        print(f"ResNet backbone initialized: {model_name}")
        print(f"Feature dimension: {feature_dim} -> {output_dim}")
    
    def forward(self, x):
        features = self.backbone(x)
        
        if self.projection is not None:
            features = self.projection(features)
            
        return features


class MambaNeck(nn.Module):
    """Mamba 기반 Feature Neck (간단화된 버전)"""
    
    def __init__(self, in_dim=768, out_dim=768, hidden_dim=512):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 간단한 MLP 기반 변환
        self.neck = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, x):
        return self.neck(x)


class VisionEncoder(nn.Module):
    """통합 Vision Encoder"""
    
    def __init__(self, 
                 model_name='vmamba_tiny_s1l8',
                 pretrained_path=None,
                 output_dim=768,
                 frozen_stages=-1,
                 use_neck=True):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.frozen_stages = frozen_stages
        
        # 백본 선택
        if VMAMBA_AVAILABLE and model_name.startswith('vmamba'):
            print("🔥 Using VMamba backbone")
            self.backbone = VMambaBackbone(
                model_name=model_name,
                pretrained=False,  # 내부 다운로드 방지
                output_dim=output_dim if not use_neck else 768
            )
            backbone_dim = 768
            
            # 사전 학습된 가중치 로딩 (VMambaBackbone에서 직접 처리)
            if pretrained_path and os.path.exists(pretrained_path):
                self.backbone.load_pretrained_weights(pretrained_path)
        else:
            print("⚠️ VMamba not available, using ResNet backbone")
            resnet_name = 'resnet50' if 'base' in model_name else 'resnet18'
            self.backbone = ResNetBackbone(
                model_name=resnet_name,
                pretrained=True,
                output_dim=output_dim if not use_neck else 768
            )
            backbone_dim = 768
        
        # Neck 추가 (선택적)
        if use_neck:
            self.neck = MambaNeck(
                in_dim=backbone_dim,
                out_dim=output_dim,
                hidden_dim=512
            )
        else:
            self.neck = None
        
        # 사전 학습된 가중치 로딩 (VMamba는 이미 위에서 처리됨)
        print(f"🔍 Checking pretrained weights...")
        if pretrained_path:
            print(f"   Pretrained path specified: {pretrained_path}")
            if os.path.exists(pretrained_path):
                if not (VMAMBA_AVAILABLE and model_name.startswith('vmamba')):
                    # VMamba가 아닌 경우에만 일반적인 가중치 로딩 수행
                    print(f"   ✅ Pretrained file found, loading...")
                    self.load_pretrained_weights(pretrained_path)
                else:
                    print(f"   ✅ VMamba weights already loaded by backbone")
            else:
                print(f"   ❌ Pretrained file not found: {pretrained_path}")
                print(f"   🔄 Using default initialization")
        else:
            print(f"   ℹ️ No pretrained path specified, using default weights")
        
        # 일부 레이어 고정
        if frozen_stages >= 0:
            self.freeze_stages(frozen_stages)
    
    def load_pretrained_weights(self, path):
        """사전 학습된 가중치 로딩"""
        print(f"📥 Loading pretrained weights from: {path}")
        try:
            print(f"   📂 Reading checkpoint file...")
            checkpoint = torch.load(path, map_location='cpu')
            print(f"   ✅ Checkpoint loaded successfully")
            
            # 체크포인트 구조 확인
            print(f"   🔍 Analyzing checkpoint structure...")
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"   📦 Using 'model' key from checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"   📦 Using 'state_dict' key from checkpoint")
            else:
                state_dict = checkpoint
                print(f"   📦 Using checkpoint directly as state_dict")
            
            print(f"   📊 Checkpoint contains {len(state_dict)} parameters")
            
            # 호환되는 키만 로딩
            print(f"   🔄 Filtering compatible parameters...")
            model_dict = self.state_dict()
            filtered_dict = {}
            
            compatible_count = 0
            incompatible_count = 0
            
            for k, v in state_dict.items():
                # VMamba 체크포인트는 backbone. 접두사가 없으므로 추가
                target_key = f"backbone.{k}"
                found = False
                
                if target_key in model_dict:
                    if model_dict[target_key].shape == v.shape:
                        filtered_dict[target_key] = v
                        compatible_count += 1
                        found = True
                    else:
                        print(f"   ⚠️ Shape mismatch for {target_key}: model={model_dict[target_key].shape} vs checkpoint={v.shape}")
                        incompatible_count += 1
                        found = True
                
                # 원본 키도 시도 (혹시 모를 경우)
                if not found:
                    if k in model_dict:
                        if model_dict[k].shape == v.shape:
                            filtered_dict[k] = v
                            compatible_count += 1
                            found = True
                        else:
                            print(f"   ⚠️ Shape mismatch for {k}: model={model_dict[k].shape} vs checkpoint={v.shape}")
                            incompatible_count += 1
                            found = True
                
                if not found:
                    print(f"   ⚠️ Key not found in model: {k} (tried: backbone.{k}, {k})")
                    incompatible_count += 1
            
            print(f"   📈 Parameter matching results:")
            print(f"      Compatible: {compatible_count}")
            print(f"      Incompatible: {incompatible_count}")
            print(f"      Total in checkpoint: {len(state_dict)}")
            print(f"      Match rate: {compatible_count/len(state_dict)*100:.1f}%")
            
            # 실제 로딩
            if compatible_count > 0:
                model_dict.update(filtered_dict)
                self.load_state_dict(model_dict, strict=False)
                print(f"   ✅ Successfully loaded {compatible_count} parameters")
            else:
                print(f"   ❌ No compatible parameters found, using random initialization")
            
        except Exception as e:
            print(f"   ❌ Failed to load pretrained weights: {e}")
            import traceback
            traceback.print_exc()
    
    def freeze_stages(self, frozen_stages):
        """지정된 스테이지까지 가중치 고정"""
        if frozen_stages >= 0:
            # 백본의 초기 레이어들 고정
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
            print(f"🧊 Frozen backbone parameters")
    
    def forward(self, x):
        # 입력이 이미지 경로인 경우 처리
        if isinstance(x, str):
            x = self.preprocess_image(x)
        
        # 백본을 통한 피처 추출
        features = self.backbone(x)
        
        # Neck 적용 (있다면)
        if self.neck is not None:
            features = self.neck(features)
        
        return features
    
    def preprocess_image(self, image_path):
        """이미지 경로를 텐서로 변환"""
        try:
            # 이미지 로딩 및 전처리
            image = Image.open(image_path).convert("RGB")
            
            # 전처리 transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
            
            # GPU로 이동 (가능한 경우)
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            # 더미 텐서 반환
            dummy = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            return dummy


def load_vision_encoder(model_name='vmamba_tiny_s1l8', **kwargs):
    """Vision Encoder 로딩 함수"""
    return VisionEncoder(model_name=model_name, **kwargs)


# 사용 가능한 모델들
AVAILABLE_MODELS = {
    'vmamba_tiny_s1l8': {
        'description': 'VMamba Tiny model with s1l8 configuration',
        'pretrained_url': 'https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth',
    },
    'resnet18': {
        'description': 'ResNet-18 backbone (fallback)',
    },
    'resnet50': {
        'description': 'ResNet-50 backbone (fallback)',
    },
}

def list_available_models():
    """사용 가능한 모델 목록 출력"""
    print("📋 Available Vision Encoder models:")
    print("-" * 50)
    for model_name, info in AVAILABLE_MODELS.items():
        status = "✅" if (VMAMBA_AVAILABLE and 'vmamba' in model_name) or ('resnet' in model_name and RESNET_AVAILABLE) else "❌"
        print(f"{status} {model_name}")
        print(f"   - {info['description']}")
        if 'pretrained_url' in info:
            print(f"   - Pretrained: {info['pretrained_url']}")
        print()


if __name__ == "__main__":
    # 사용 예시
    list_available_models()
    
    # Vision Encoder 테스트
    encoder = load_vision_encoder()
    
    # 더미 이미지로 테스트
    dummy_image = torch.randn(1, 3, 224, 224)
    features = encoder(dummy_image)
    print(f"Vision features shape: {features.shape}")
    print(f"Features norm: {features.norm()}") 