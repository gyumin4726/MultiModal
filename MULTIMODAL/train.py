import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MultiModalVQAClassifier
from datasets import VQADataLoader
from configs import get_config
from utils import setup_logging, save_checkpoint, load_checkpoint, AverageMeter


def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model: nn.Module, config):
    """옵티마이저 생성"""
    if config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def create_scheduler(optimizer, config, num_training_steps):
    """스케줄러 생성"""
    if config.scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.learning_rate * 0.01
        )
    elif config.scheduler.lower() == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=num_training_steps
        )
    elif config.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_training_steps // 3,
            gamma=0.1
        )
    else:
        scheduler = None
    
    return scheduler


def create_criterion(config):
    """손실 함수 생성"""
    if config.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    return criterion


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, config, epoch):
    """한 에포크 학습"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        # 데이터를 디바이스로 이동
        images = batch['images'].to(device, non_blocking=True)
        answers = batch['answers'].to(device, non_blocking=True)
        
        # 텍스트 데이터
        questions = batch['questions']
        choices_a = batch['choices_a']
        choices_b = batch['choices_b']
        choices_c = batch['choices_c']
        choices_d = batch['choices_d']
        
        batch_size = images.size(0)
        
        # 순전파
        with autocast(enabled=config.mixed_precision):
            logits = model(
                images, questions, choices_a, choices_b, choices_c, choices_d,
                encoding_mode='unified'
            )
            loss = criterion(logits, answers)
        
        # 역전파
        optimizer.zero_grad()
        if config.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # 정확도 계산
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == answers).float().mean()
        
        # 메트릭 업데이트
        losses.update(loss.item(), batch_size)
        accuracies.update(accuracy.item(), batch_size)
        
        # 진행 상황 업데이트
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accuracies.avg:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 로깅
        if batch_idx % config.log_every == 0:
            step = epoch * len(dataloader) + batch_idx
            wandb.log({
                'train/loss': loss.item(),
                'train/accuracy': accuracy.item(),
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/step': step
            })
    
    return losses.avg, accuracies.avg


def validate_epoch(model, dataloader, criterion, device, config, epoch):
    """한 에포크 검증"""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_predictions = []
    all_answers = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation {epoch+1}")
        
        for batch in pbar:
            # 데이터를 디바이스로 이동
            images = batch['images'].to(device, non_blocking=True)
            answers = batch['answers'].to(device, non_blocking=True)
            
            # 텍스트 데이터
            questions = batch['questions']
            choices_a = batch['choices_a']
            choices_b = batch['choices_b']
            choices_c = batch['choices_c']
            choices_d = batch['choices_d']
            
            batch_size = images.size(0)
            
            # 순전파
            with autocast(enabled=config.mixed_precision):
                logits = model(
                    images, questions, choices_a, choices_b, choices_c, choices_d,
                    encoding_mode='unified'
                )
                loss = criterion(logits, answers)
            
            # 정확도 계산
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == answers).float().mean()
            
            # 메트릭 업데이트
            losses.update(loss.item(), batch_size)
            accuracies.update(accuracy.item(), batch_size)
            
            # 예측 결과 저장
            all_predictions.extend(predictions.cpu().numpy())
            all_answers.extend(answers.cpu().numpy())
            
            # 진행 상황 업데이트
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.4f}'
            })
    
    # 클래스별 정확도 계산
    class_accuracies = {}
    for class_idx in range(4):
        class_mask = np.array(all_answers) == class_idx
        if class_mask.sum() > 0:
            class_acc = (np.array(all_predictions)[class_mask] == class_idx).mean()
            class_accuracies[f'class_{class_idx}'] = class_acc
    
    return losses.avg, accuracies.avg, class_accuracies


def main():
    parser = argparse.ArgumentParser(description='Train MultiModal VQA Model')
    parser.add_argument('--config', type=str, default='competition',
                       choices=['default', 'tiny', 'large', 'competition'],
                       help='Configuration type')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--wandb_project', type=str, default='scpc-vqa',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (no wandb logging)')
    
    args = parser.parse_args()
    
    # 설정 로드
    model_config, training_config, _ = get_config(args.config)
    
    # 시드 설정
    set_seed(training_config.seed)
    
    # 디바이스 설정
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 디렉토리 생성
    os.makedirs(training_config.save_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)
    
    # 로깅 설정
    logger = setup_logging(
        log_file=os.path.join(training_config.log_dir, 'train.log')
    )
    
    # Weights & Biases 초기화
    if not args.debug:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                'model_config': model_config.__dict__,
                'training_config': training_config.__dict__
            }
        )
    
    # 모델 생성
    logger.info("Creating model...")
    model = MultiModalVQAClassifier(
        image_encoder_config=model_config.image_encoder,
        text_encoder_config=model_config.text_encoder,
        fusion_config=model_config.fusion,
        num_classes=model_config.num_classes,
        dropout=model_config.dropout,
        use_adaptive_fusion=model_config.use_adaptive_fusion
    )
    model = model.to(device)
    
    # 모델 컴파일 (PyTorch 2.0+)
    if training_config.compile_model and hasattr(torch, 'compile'):
        logger.info("Compiling model...")
        model = torch.compile(model)
    
    # 데이터로더 생성
    logger.info("Creating data loaders...")
    train_loader = VQADataLoader.create_train_loader(
        csv_path=training_config.train_csv,
        image_dir=training_config.train_image_dir,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        shuffle=True
    )
    
    val_loader = VQADataLoader.create_test_loader(
        csv_path=training_config.val_csv,
        image_dir=training_config.val_image_dir,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers
    )
    
    # 옵티마이저, 스케줄러, 손실함수 생성
    num_training_steps = len(train_loader) * training_config.num_epochs
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config, num_training_steps)
    criterion = create_criterion(training_config)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=training_config.mixed_precision)
    
    # 체크포인트 로드
    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # 학습 루프
    logger.info("Starting training...")
    for epoch in range(start_epoch, training_config.num_epochs):
        
        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, training_config, epoch
        )
        
        logger.info(f"Epoch {epoch+1}/{training_config.num_epochs}")
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # 검증
        if (epoch + 1) % training_config.val_every == 0:
            val_loss, val_acc, class_accs = validate_epoch(
                model, val_loader, criterion, device, training_config, epoch
            )
            
            logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            logger.info(f"Class accuracies: {class_accs}")
            
            # Weights & Biases 로깅
            if not args.debug:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'train/epoch_accuracy': train_acc,
                    'val/epoch_loss': val_loss,
                    'val/epoch_accuracy': val_acc
                }
                log_dict.update({f'val/{k}': v for k, v in class_accs.items()})
                wandb.log(log_dict)
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_acc,
                    os.path.join(training_config.save_dir, 'best_model.pth')
                )
                logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
        
        # 정기적으로 체크포인트 저장
        if (epoch + 1) % training_config.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_acc,
                os.path.join(training_config.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # 조기 종료
        if patience_counter >= training_config.patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # 최종 모델 저장
    save_checkpoint(
        model, optimizer, scheduler, epoch, val_acc,
        os.path.join(training_config.save_dir, 'final_model.pth')
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    main() 