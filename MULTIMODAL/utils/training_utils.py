import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import time


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """로깅 설정
    
    Args:
        log_file (str, optional): 로그 파일 경로
        level (int): 로깅 레벨
        
    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    val_acc: float,
    filepath: str,
    **kwargs
):
    """체크포인트 저장
    
    Args:
        model (nn.Module): 모델
        optimizer (torch.optim.Optimizer): 옵티마이저
        scheduler: 스케줄러
        epoch (int): 에포크
        val_acc (float): 검증 정확도
        filepath (str): 저장 경로
        **kwargs: 추가 정보
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': val_acc,
        'timestamp': time.time(),
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, device: str = 'cpu') -> Dict[str, Any]:
    """체크포인트 로드
    
    Args:
        filepath (str): 체크포인트 파일 경로
        device (str): 디바이스
        
    Returns:
        Dict[str, Any]: 체크포인트 데이터
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    
    return checkpoint


class AverageMeter:
    """평균 메트릭 계산기"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """조기 종료 클래스"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Args:
            score (float): 현재 점수 (높을수록 좋음)
            model (nn.Module): 모델
            
        Returns:
            bool: 조기 종료 여부
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """최고 성능 모델 저장"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


class MetricTracker:
    """메트릭 추적기"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """메트릭 업데이트"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
            self.metrics[key].update(value)
    
    def get_averages(self) -> Dict[str, float]:
        """평균 메트릭 반환"""
        return {key: meter.avg for key, meter in self.metrics.items()}
    
    def reset(self):
        """모든 메트릭 리셋"""
        for meter in self.metrics.values():
            meter.reset()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """모델 파라미터 수 계산
    
    Args:
        model (nn.Module): 모델
        
    Returns:
        Dict[str, int]: 파라미터 정보
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> float:
    """모델 크기 계산 (MB)
    
    Args:
        model (nn.Module): 모델
        
    Returns:
        float: 모델 크기 (MB)
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(model: nn.Module):
    """모델 정보 출력"""
    param_info = count_parameters(model)
    size_mb = get_model_size(model)
    
    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print(f"Non-trainable parameters: {param_info['non_trainable']:,}")
    print(f"Model size: {size_mb:.2f} MB")
    
    # 3B 파라미터 제한 확인
    if param_info['total'] >= 3_000_000_000:
        print("⚠️  WARNING: Model exceeds 3B parameter limit!")
    else:
        print("✅ Model is within 3B parameter limit")
    
    print("=" * 50)


class WarmupScheduler:
    """워밍업 스케줄러"""
    
    def __init__(self, optimizer, warmup_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
    
    def step(self):
        """스케줄러 스텝"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # 워밍업 단계
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # 정상 학습률
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """현재 학습률 반환"""
        return self.optimizer.param_groups[0]['lr']


if __name__ == "__main__":
    # 유틸리티 함수 테스트
    import torch.nn as nn
    
    # 더미 모델
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    print_model_info(model)
    
    # AverageMeter 테스트
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    # EarlyStopping 테스트
    early_stopping = EarlyStopping(patience=3)
    for epoch in range(10):
        score = 0.9 - epoch * 0.01  # 점수가 감소
        if early_stopping(score, model):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("Training utils test completed!") 