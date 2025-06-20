import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.zero_shot_classifier import ZeroShotVQAClassifier
from datasets import VQADataLoader
from utils import setup_logging


def load_test_data(csv_path: str, image_dir: str):
    """테스트 데이터 로드"""
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} test samples from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def create_zero_shot_model(device: str = 'cuda'):
    """Zero-shot 모델 생성"""
    
    # 모델 설정 (train_vmamba_fscil.sh와 동일한 백본 사용)
    image_encoder_config = {
        'model_name': 'vmamba_tiny_s1l8',
        'pretrained_path': './vssm1_tiny_0230s_ckpt_epoch_264.pth',
        'output_dim': 768,
        'frozen_stages': 1,
        'out_indices': (3,),
        'channel_first': True,
        'image_size': 224
    }
    
    text_encoder_config = {
        'model_name': 'bert-base-uncased',
        'output_dim': 768,
        'max_length': 512,
        'freeze_bert': True  # Zero-shot이므로 고정
    }
    
    print("Creating Zero-shot VQA Classifier...")
    model = ZeroShotVQAClassifier(
        image_encoder_config=image_encoder_config,
        text_encoder_config=text_encoder_config,
        similarity_method='cosine',
        temperature=1.0
    )
    
    model = model.to(device)
    model.eval()
    
    return model


def run_zero_shot_inference(model, df, image_dir, device, batch_size=8):
    """Zero-shot 추론 실행"""
    
    # 데이터로더 생성
    dataloader = VQADataLoader.create_test_loader(
        csv_path=None,  # 직접 DataFrame 사용
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=4,
        df=df  # DataFrame 직접 전달
    )
    
    all_predictions = []
    all_confidences = []
    all_probabilities = []
    
    print("Running zero-shot inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            # 데이터를 디바이스로 이동
            images = batch['images'].to(device, non_blocking=True)
            
            # 텍스트 데이터 추출
            questions = batch['questions']
            choices_a = batch['choices_a']
            choices_b = batch['choices_b']
            choices_c = batch['choices_c']
            choices_d = batch['choices_d']
            
            # 신뢰도와 함께 예측
            detailed_results = model.predict_with_confidence(
                images, questions, choices_a, choices_b, choices_c, choices_d
            )
            
            # 결과 수집
            for result in detailed_results:
                all_predictions.append(result['answer'])
                all_confidences.append(result['confidence'])
                all_probabilities.append(result['probabilities'])
    
    return all_predictions, all_confidences, all_probabilities


def save_submission(df, predictions, confidences, probabilities, output_path):
    """제출 파일 저장"""
    
    # 기본 제출 파일
    submission_df = pd.DataFrame({
        'ID': df['ID'],
        'answer': predictions
    })
    
    # 상세 결과 파일 (분석용)
    detailed_df = df.copy()
    detailed_df['predicted_answer'] = predictions
    detailed_df['confidence'] = confidences
    
    # 각 선택지별 확률 추가
    for i, choice in enumerate(['A', 'B', 'C', 'D']):
        detailed_df[f'prob_{choice}'] = [prob[choice] for prob in probabilities]
    
    # 파일 저장
    base_name = os.path.splitext(output_path)[0]
    
    submission_df.to_csv(output_path, index=False)
    detailed_df.to_csv(f"{base_name}_detailed.csv", index=False)
    
    print(f"Submission saved to: {output_path}")
    print(f"Detailed results saved to: {base_name}_detailed.csv")
    
    # 간단한 통계 출력
    print(f"\nPrediction Statistics:")
    print(f"Answer distribution:")
    print(submission_df['answer'].value_counts().sort_index())
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Min confidence: {np.min(confidences):.4f}")
    print(f"Max confidence: {np.max(confidences):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Zero-shot VQA Inference')
    parser.add_argument('--test_csv', type=str, default='dev/dev_test.csv',
                       help='Test CSV file path')
    parser.add_argument('--image_dir', type=str, default='dev/input_images',
                       help='Test images directory')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output submission file path')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--similarity_method', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'dot'],
                       help='Similarity computation method')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for softmax scaling')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 로깅 설정
    logger = setup_logging(log_file='zero_shot_inference.log')
    
    # 테스트 데이터 로드
    logger.info(f"Loading test data from {args.test_csv}")
    df = load_test_data(args.test_csv, args.image_dir)
    
    # 모델 생성
    logger.info("Creating zero-shot model...")
    model = create_zero_shot_model(device)
    
    # 설정 업데이트
    model.similarity_method = args.similarity_method
    model.temperature = args.temperature
    
    logger.info(f"Model configuration:")
    logger.info(f"  Similarity method: {args.similarity_method}")
    logger.info(f"  Temperature: {args.temperature}")
    
    # 추론 실행
    logger.info("Starting zero-shot inference...")
    predictions, confidences, probabilities = run_zero_shot_inference(
        model, df, args.image_dir, device, args.batch_size
    )
    
    # 결과 저장
    logger.info("Saving results...")
    save_submission(df, predictions, confidences, probabilities, args.output)
    
    logger.info("Zero-shot inference completed!")
    
    # 샘플 결과 출력
    print(f"\nSample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"ID: {df.iloc[i]['ID']}")
        print(f"Question: {df.iloc[i]['Question']}")
        print(f"Choices: A) {df.iloc[i]['A']}, B) {df.iloc[i]['B']}, C) {df.iloc[i]['C']}, D) {df.iloc[i]['D']}")
        print(f"Predicted: {predictions[i]} (confidence: {confidences[i]:.4f})")
        print(f"Probabilities: {probabilities[i]}")
        print("-" * 80)


if __name__ == "__main__":
    main() 