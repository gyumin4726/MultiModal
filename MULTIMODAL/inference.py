import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MultiModalVQAClassifier
from datasets import VQADataLoader
from configs import get_config
from utils import load_checkpoint
from utils.submission_utils import create_baseline_submission, validate_submission_format


def predict_batch(model, batch, device, encoding_mode='unified'):
    """배치 예측"""
    model.eval()
    
    # 데이터를 디바이스로 이동
    images = batch['images'].to(device, non_blocking=True)
    
    # 텍스트 데이터
    questions = batch['questions']
    choices_a = batch['choices_a']
    choices_b = batch['choices_b']
    choices_c = batch['choices_c']
    choices_d = batch['choices_d']
    ids = batch['ids']
    
    with torch.no_grad():
        # 예측
        predictions = model.predict(
            images, questions, choices_a, choices_b, choices_c, choices_d,
            encoding_mode=encoding_mode
        )
    
    return predictions, ids


def run_inference(model, dataloader, device, encoding_mode='unified'):
    """전체 데이터셋에 대한 추론"""
    model.eval()
    
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference")
        
        for batch in pbar:
            predictions, ids = predict_batch(model, batch, device, encoding_mode)
            
            all_predictions.extend(predictions)
            all_ids.extend(ids)
            
            pbar.set_postfix({'Processed': len(all_ids)})
    
    return all_predictions, all_ids


def create_submission(predictions, ids, output_path, sample_submission_path='./sample_submission.csv'):
    """제출 파일 생성"""
    # sample_submission.csv 파일을 기반으로 생성
    if os.path.exists(sample_submission_path):
        submission = pd.read_csv(sample_submission_path)
        print(f"Using sample submission template: {sample_submission_path}")
    else:
        # sample_submission.csv가 없으면 직접 생성
        print(f"Sample submission not found at {sample_submission_path}, creating from scratch")
        submission = pd.DataFrame({
            'ID': ids,
            'answer': 'A'  # 기본값
        })
    
    # ID와 예측 결과를 매칭
    id_to_prediction = dict(zip(ids, predictions))
    
    # submission 파일의 각 ID에 대해 예측 결과 할당
    for idx, row in submission.iterrows():
        sample_id = row['ID']
        if sample_id in id_to_prediction:
            submission.at[idx, 'answer'] = id_to_prediction[sample_id]
        else:
            print(f"Warning: No prediction found for ID {sample_id}, keeping default answer")
    
    # CSV 파일로 저장
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    print(f"Total predictions: {len(submission)}")
    
    # 예측 분포 출력
    answer_dist = submission['answer'].value_counts().sort_index()
    print(f"Answer distribution:")
    for answer, count in answer_dist.items():
        print(f"  {answer}: {count} ({count/len(submission)*100:.1f}%)")
    
    return submission


def main():
    parser = argparse.ArgumentParser(description='Inference for MultiModal VQA Model')
    parser.add_argument('--config', type=str, default='competition',
                       choices=['default', 'tiny', 'large', 'competition'],
                       help='Configuration type')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, default='data/dev_test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--test_image_dir', type=str, default='data/input_images',
                       help='Path to test images directory')
    parser.add_argument('--output_csv', type=str, default='baseline_submit.csv',
                       help='Path to output submission CSV file')
    parser.add_argument('--sample_submission', type=str, default='./sample_submission.csv',
                       help='Path to sample submission CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--encoding_mode', type=str, default='unified',
                       choices=['unified', 'joint', 'separate'],
                       help='Text encoding mode')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # 설정 로드
    model_config, _, inference_config = get_config(args.config)
    
    # 명령행 인수로 설정 오버라이드
    if args.test_csv:
        inference_config.test_csv = args.test_csv
    if args.test_image_dir:
        inference_config.test_image_dir = args.test_image_dir
    if args.output_csv:
        inference_config.output_csv = args.output_csv
    if args.batch_size:
        inference_config.batch_size = args.batch_size
    if args.encoding_mode:
        inference_config.encoding_mode = args.encoding_mode
    if args.device:
        inference_config.device = args.device
    
    # 디바이스 설정
    device = torch.device(inference_config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 생성
    print("Creating model...")
    model = MultiModalVQAClassifier(
        image_encoder_config=model_config.image_encoder,
        text_encoder_config=model_config.text_encoder,
        fusion_config=model_config.fusion,
        num_classes=model_config.num_classes,
        dropout=model_config.dropout,
        use_adaptive_fusion=model_config.use_adaptive_fusion
    )
    
    # 체크포인트 로드
    print(f"Loading model from: {args.model_path}")
    checkpoint = load_checkpoint(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    
    # 데이터로더 생성
    print("Creating data loader...")
    test_loader = VQADataLoader.create_test_loader(
        csv_path=inference_config.test_csv,
        image_dir=inference_config.test_image_dir,
        batch_size=inference_config.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # 추론 실행
    print("Running inference...")
    predictions, ids = run_inference(
        model, test_loader, device, inference_config.encoding_mode
    )
    
    # 제출 파일 생성 (사용자 방식)
    print("Creating submission file...")
    submission_df = create_baseline_submission(
        predictions=predictions,
        ids=ids,
        sample_submission_path=args.sample_submission,
        output_path=inference_config.output_csv
    )
    
    # 제출 파일 검증
    print("\n=== Validating Submission ===")
    validate_submission_format(inference_config.output_csv)
    
    # 결과 요약
    print("\n=== Inference Results ===")
    print(f"Total samples processed: {len(predictions)}")
    print(f"Unique IDs: {len(set(ids))}")
    print(f"Output file: {inference_config.output_csv}")
    
    # 샘플 결과 출력
    print("\n=== Sample Predictions ===")
    for i in range(min(5, len(submission_df))):
        row = submission_df.iloc[i]
        print(f"ID: {row['ID']}, Answer: {row['answer']}")
    
    print("\n✅ Inference completed successfully!")


if __name__ == "__main__":
    main() 