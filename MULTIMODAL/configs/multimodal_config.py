from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    
    # 이미지 인코더 설정
    image_encoder: Dict[str, Any] = None
    
    # 텍스트 인코더 설정  
    text_encoder: Dict[str, Any] = None
    
    # 퓨전 모듈 설정
    fusion: Dict[str, Any] = None
    
    # 분류기 설정
    num_classes: int = 4
    dropout: float = 0.1
    use_adaptive_fusion: bool = False
    
    def __post_init__(self):
        if self.image_encoder is None:
            self.image_encoder = {
                'model_name': 'vmamba_tiny_s1l8',  # train_vmamba_fscil.sh와 동일
                'pretrained_path': './vssm1_tiny_0230s_ckpt_epoch_264.pth',  # train_vmamba_fscil.sh와 동일
                'output_dim': 768,
                'frozen_stages': 1,  # train_vmamba_fscil.sh와 동일
                'out_indices': (3,),  # 마지막 스테이지만 사용
                'channel_first': True,  # train_vmamba_fscil.sh와 동일
                'image_size': 224
            }
        
        if self.text_encoder is None:
            self.text_encoder = {
                'model_name': 'bert-base-uncased',
                'output_dim': 768,
                'max_length': 512,
                'freeze_bert': False
            }
        
        if self.fusion is None:
            self.fusion = {
                'hidden_dim': 768,
                'num_heads': 8,
                'fusion_method': 'cross_attention'
            }


@dataclass
class TrainingConfig:
    """학습 설정 클래스"""
    
    # 데이터 설정
    train_csv: str = "data/train.csv"
    train_image_dir: str = "data/train_images"
    val_csv: str = "data/val.csv"
    val_image_dir: str = "data/val_images"
    test_csv: str = "data/dev_test.csv"
    test_image_dir: str = "data/input_images"
    
    # 학습 하이퍼파라미터
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # 옵티마이저 설정
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, linear, step
    
    # 데이터로더 설정
    num_workers: int = 4
    pin_memory: bool = True
    
    # 정규화 및 증강
    dropout: float = 0.1
    label_smoothing: float = 0.1
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    
    # 체크포인트 및 로깅
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 5
    log_every: int = 100
    
    # 검증 및 조기 종료
    val_every: int = 1
    patience: int = 10
    min_delta: float = 1e-4
    
    # 하드웨어 설정
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = False
    
    # 재현성
    seed: int = 42
    deterministic: bool = True


@dataclass
class InferenceConfig:
    """추론 설정 클래스"""
    
    # 모델 설정
    model_path: str = "checkpoints/best_model.pth"
    
    # 데이터 설정
    test_csv: str = "data/dev_test.csv"
    test_image_dir: str = "data/input_images"
    output_csv: str = "submission.csv"
    
    # 추론 설정
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda"
    
    # 인코딩 모드
    encoding_mode: str = "unified"  # unified, joint, separate
    
    # TTA (Test Time Augmentation)
    use_tta: bool = False
    tta_transforms: int = 5


def get_config(config_type: str = "default") -> tuple:
    """설정 반환 함수
    
    Args:
        config_type (str): 설정 타입
            - "default": 기본 설정
            - "tiny": 작은 모델 설정 (빠른 실험용)
            - "large": 큰 모델 설정 (최고 성능용)
            - "competition": 대회 제출용 설정
    
    Returns:
        tuple: (ModelConfig, TrainingConfig, InferenceConfig)
    """
    
    if config_type == "default":
        model_config = ModelConfig()
        training_config = TrainingConfig()
        inference_config = InferenceConfig()
        
    elif config_type == "tiny":
        model_config = ModelConfig(
            image_encoder={
                'model_name': 'vmamba_tiny_s1l8',
                'output_dim': 384,
                'frozen_stages': 2,
                'image_size': 224
            },
            text_encoder={
                'model_name': 'bert-base-uncased',
                'output_dim': 384,
                'max_length': 256,
                'freeze_bert': True
            },
            fusion={
                'hidden_dim': 384,
                'num_heads': 6,
                'fusion_method': 'concat'
            },
            dropout=0.2
        )
        training_config = TrainingConfig(
            batch_size=32,
            num_epochs=20,
            learning_rate=2e-4,
            num_workers=2
        )
        inference_config = InferenceConfig(
            batch_size=64,
            encoding_mode="joint"
        )
        
    elif config_type == "large":
        model_config = ModelConfig(
            image_encoder={
                'model_name': 'vmamba_base_s1l20',
                'output_dim': 1024,
                'frozen_stages': 0,
                'image_size': 224
            },
            text_encoder={
                'model_name': 'bert-large-uncased',
                'output_dim': 1024,
                'max_length': 512,
                'freeze_bert': False
            },
            fusion={
                'hidden_dim': 1024,
                'num_heads': 16,
                'fusion_method': 'cross_attention'
            },
            use_adaptive_fusion=True,
            dropout=0.1
        )
        training_config = TrainingConfig(
            batch_size=8,
            num_epochs=100,
            learning_rate=5e-5,
            weight_decay=1e-4,
            use_mixup=True,
            label_smoothing=0.1
        )
        inference_config = InferenceConfig(
            batch_size=16,
            encoding_mode="unified",
            use_tta=True
        )
        
    elif config_type == "competition":
        # 대회 제출용 최적화된 설정
        model_config = ModelConfig(
            image_encoder={
                'model_name': 'vmamba_tiny_s1l8',
                'pretrained_path': './pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth',
                'output_dim': 768,
                'frozen_stages': 1,
                'image_size': 224
            },
            text_encoder={
                'model_name': 'bert-base-uncased',
                'output_dim': 768,
                'max_length': 512,
                'freeze_bert': False
            },
            fusion={
                'hidden_dim': 768,
                'num_heads': 12,
                'fusion_method': 'cross_attention'
            },
            use_adaptive_fusion=False,
            dropout=0.1
        )
        training_config = TrainingConfig(
            batch_size=16,
            num_epochs=80,
            learning_rate=1e-4,
            weight_decay=1e-5,
            warmup_steps=2000,
            use_mixup=True,
            mixup_alpha=0.1,
            label_smoothing=0.05,
            patience=15,
            mixed_precision=True
        )
        inference_config = InferenceConfig(
            batch_size=32,
            encoding_mode="unified",
            use_tta=False
        )
        
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return model_config, training_config, inference_config


def validate_config(model_config: ModelConfig, training_config: TrainingConfig):
    """설정 유효성 검사"""
    
    # 모델 파라미터 수 추정 (간단한 계산)
    image_dim = model_config.image_encoder['output_dim']
    text_dim = model_config.text_encoder['output_dim']
    fusion_dim = model_config.fusion['hidden_dim']
    
    # VMamba 파라미터 추정
    if 'tiny' in model_config.image_encoder['model_name']:
        vmamba_params = 22_000_000  # 약 22M
    elif 'small' in model_config.image_encoder['model_name']:
        vmamba_params = 44_000_000  # 약 44M
    elif 'base' in model_config.image_encoder['model_name']:
        vmamba_params = 75_000_000  # 약 75M
    else:
        vmamba_params = 22_000_000  # 기본값
    
    # BERT 파라미터 추정
    if 'large' in model_config.text_encoder['model_name']:
        bert_params = 340_000_000  # 약 340M
    else:
        bert_params = 110_000_000  # 약 110M (base)
    
    # 퓨전 및 분류기 파라미터 추정
    fusion_params = fusion_dim * (image_dim + text_dim + fusion_dim * 4)
    classifier_params = fusion_dim * (fusion_dim // 2 + fusion_dim // 4 + 4)
    
    total_params = vmamba_params + bert_params + fusion_params + classifier_params
    
    print(f"Estimated model parameters:")
    print(f"  VMamba: {vmamba_params:,}")
    print(f"  BERT: {bert_params:,}")
    print(f"  Fusion: {fusion_params:,}")
    print(f"  Classifier: {classifier_params:,}")
    print(f"  Total: {total_params:,}")
    
    # 3B 제한 확인
    if total_params >= 3_000_000_000:
        print("WARNING: Model may exceed 3B parameter limit!")
        return False
    
    # 배치 크기와 메모리 확인
    if training_config.batch_size > 32 and 'large' in model_config.text_encoder['model_name']:
        print("WARNING: Large batch size with large model may cause OOM!")
    
    return True


if __name__ == "__main__":
    # 설정 테스트
    for config_type in ["default", "tiny", "large", "competition"]:
        print(f"\n=== {config_type.upper()} CONFIG ===")
        model_cfg, train_cfg, infer_cfg = get_config(config_type)
        
        print(f"Model config: {model_cfg}")
        print(f"Training config: {train_cfg}")
        print(f"Inference config: {infer_cfg}")
        
        validate_config(model_cfg, train_cfg) 