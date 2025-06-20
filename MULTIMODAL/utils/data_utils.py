import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import torch
from collections import Counter


def create_data_splits(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """데이터를 train/val/test로 분할
    
    Args:
        csv_path (str): CSV 파일 경로
        test_size (float): 테스트 데이터 비율
        val_size (float): 검증 데이터 비율
        random_state (int): 랜덤 시드
        stratify_column (str): 계층화 기준 컬럼
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test 데이터프레임
    """
    df = pd.read_csv(csv_path)
    
    stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
    
    # train + val, test 분할
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # train, val 분할
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        stratify_train_val = train_val[stratify_column] if stratify_column and stratify_column in train_val.columns else None
        
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state, stratify=stratify_train_val
        )
    else:
        train = train_val
        val = pd.DataFrame()
    
    print(f"Data split completed:")
    print(f"  Train: {len(train)} samples")
    print(f"  Val: {len(val)} samples")
    print(f"  Test: {len(test)} samples")
    
    return train, val, test


def analyze_dataset(csv_path: str, image_dir: str = None) -> Dict:
    """데이터셋 분석
    
    Args:
        csv_path (str): CSV 파일 경로
        image_dir (str): 이미지 디렉토리 경로
        
    Returns:
        Dict: 분석 결과
    """
    df = pd.read_csv(csv_path)
    
    analysis = {
        'basic_stats': {
            'total_samples': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict()
        }
    }
    
    # 텍스트 분석
    if 'Question' in df.columns:
        questions = df['Question'].astype(str)
        analysis['question_stats'] = {
            'unique_questions': questions.nunique(),
            'avg_length': questions.str.len().mean(),
            'max_length': questions.str.len().max(),
            'min_length': questions.str.len().min(),
            'word_count_avg': questions.str.split().str.len().mean()
        }
    
    # 선택지 분석
    choice_cols = ['A', 'B', 'C', 'D']
    if all(col in df.columns for col in choice_cols):
        choice_stats = {}
        for col in choice_cols:
            choices = df[col].astype(str)
            choice_stats[col] = {
                'avg_length': choices.str.len().mean(),
                'max_length': choices.str.len().max(),
                'unique_choices': choices.nunique()
            }
        analysis['choice_stats'] = choice_stats
    
    # 정답 분포
    if 'answer' in df.columns:
        answer_dist = df['answer'].value_counts().to_dict()
        analysis['answer_distribution'] = answer_dist
    
    # 이미지 분석
    if image_dir and os.path.exists(image_dir):
        image_analysis = analyze_images(df, image_dir)
        analysis['image_stats'] = image_analysis
    
    return analysis


def analyze_images(df: pd.DataFrame, image_dir: str, sample_size: int = 100) -> Dict:
    """이미지 분석
    
    Args:
        df (pd.DataFrame): 데이터프레임
        image_dir (str): 이미지 디렉토리
        sample_size (int): 분석할 샘플 수
        
    Returns:
        Dict: 이미지 분석 결과
    """
    if 'img_path' not in df.columns:
        return {}
    
    # 샘플링
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    image_sizes = []
    image_modes = []
    file_sizes = []
    
    for _, row in sample_df.iterrows():
        img_path = os.path.join(image_dir, row['img_path'])
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    image_sizes.append(img.size)
                    image_modes.append(img.mode)
                file_sizes.append(os.path.getsize(img_path))
            except Exception as e:
                print(f"Error analyzing image {img_path}: {e}")
    
    if not image_sizes:
        return {'error': 'No valid images found'}
    
    widths, heights = zip(*image_sizes)
    
    return {
        'sample_count': len(image_sizes),
        'size_stats': {
            'width': {'mean': np.mean(widths), 'std': np.std(widths), 'min': min(widths), 'max': max(widths)},
            'height': {'mean': np.mean(heights), 'std': np.std(heights), 'min': min(heights), 'max': max(heights)}
        },
        'modes': dict(Counter(image_modes)),
        'file_size_mb': {
            'mean': np.mean(file_sizes) / (1024*1024),
            'std': np.std(file_sizes) / (1024*1024),
            'total_gb': sum(file_sizes) / (1024*1024*1024)
        }
    }


def visualize_samples(df: pd.DataFrame, image_dir: str, num_samples: int = 6, figsize: Tuple[int, int] = (15, 10)):
    """샘플 데이터 시각화
    
    Args:
        df (pd.DataFrame): 데이터프레임
        image_dir (str): 이미지 디렉토리
        num_samples (int): 시각화할 샘플 수
        figsize (Tuple[int, int]): 그래프 크기
    """
    sample_df = df.sample(num_samples, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(sample_df.iterrows()):
        if i >= num_samples:
            break
            
        ax = axes[i]
        
        # 이미지 로드 및 표시
        img_path = os.path.join(image_dir, row['img_path'])
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis('off')
                
                # 제목 설정
                question = row['Question'][:50] + "..." if len(row['Question']) > 50 else row['Question']
                choices = f"A:{row['A'][:10]}... B:{row['B'][:10]}..."
                answer = f"Answer: {row.get('answer', 'N/A')}"
                
                title = f"{question}\n{choices}\n{answer}"
                ax.set_title(title, fontsize=8, wrap=True)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{img_path}", ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"Image not found\n{img_path}", ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_class_weights(df: pd.DataFrame, answer_column: str = 'answer') -> Dict[str, float]:
    """클래스 가중치 계산
    
    Args:
        df (pd.DataFrame): 데이터프레임
        answer_column (str): 정답 컬럼명
        
    Returns:
        Dict[str, float]: 클래스별 가중치
    """
    if answer_column not in df.columns:
        return {}
    
    answer_counts = df[answer_column].value_counts()
    total_samples = len(df)
    num_classes = len(answer_counts)
    
    # 역빈도 가중치 계산
    class_weights = {}
    for answer, count in answer_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[answer] = weight
    
    return class_weights


def create_submission_template(test_csv: str, output_path: str = 'submission_template.csv'):
    """제출 파일 템플릿 생성
    
    Args:
        test_csv (str): 테스트 CSV 파일 경로
        output_path (str): 출력 파일 경로
    """
    df = pd.read_csv(test_csv)
    
    if 'ID' not in df.columns:
        raise ValueError("Test CSV must contain 'ID' column")
    
    submission = pd.DataFrame({
        'ID': df['ID'],
        'answer': 'A'  # 기본값
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission template created: {output_path}")


def validate_submission(submission_path: str, test_csv: str) -> bool:
    """제출 파일 유효성 검사
    
    Args:
        submission_path (str): 제출 파일 경로
        test_csv (str): 테스트 CSV 파일 경로
        
    Returns:
        bool: 유효성 여부
    """
    try:
        # 제출 파일 로드
        submission = pd.read_csv(submission_path)
        test_df = pd.read_csv(test_csv)
        
        # 필수 컬럼 확인
        if not all(col in submission.columns for col in ['ID', 'answer']):
            print("❌ Missing required columns: ID, answer")
            return False
        
        # ID 일치 확인
        if set(submission['ID']) != set(test_df['ID']):
            print("❌ ID mismatch between submission and test data")
            return False
        
        # 답변 형식 확인
        valid_answers = {'A', 'B', 'C', 'D'}
        if not submission['answer'].isin(valid_answers).all():
            print("❌ Invalid answers found. Only A, B, C, D are allowed")
            return False
        
        # 중복 ID 확인
        if submission['ID'].duplicated().any():
            print("❌ Duplicate IDs found in submission")
            return False
        
        print("✅ Submission file is valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error validating submission: {e}")
        return False


def plot_dataset_statistics(analysis: Dict, save_path: str = None):
    """데이터셋 통계 시각화
    
    Args:
        analysis (Dict): analyze_dataset 결과
        save_path (str): 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 정답 분포
    if 'answer_distribution' in analysis:
        ax = axes[0, 0]
        answer_dist = analysis['answer_distribution']
        ax.bar(answer_dist.keys(), answer_dist.values())
        ax.set_title('Answer Distribution')
        ax.set_xlabel('Answer')
        ax.set_ylabel('Count')
    
    # 질문 길이 분포
    if 'question_stats' in analysis:
        ax = axes[0, 1]
        # 이 부분은 실제 데이터가 있어야 히스토그램을 그릴 수 있음
        ax.text(0.5, 0.5, f"Avg Question Length: {analysis['question_stats']['avg_length']:.1f}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Question Statistics')
    
    # 선택지 길이 비교
    if 'choice_stats' in analysis:
        ax = axes[1, 0]
        choice_lengths = [stats['avg_length'] for stats in analysis['choice_stats'].values()]
        choice_labels = list(analysis['choice_stats'].keys())
        ax.bar(choice_labels, choice_lengths)
        ax.set_title('Average Choice Length')
        ax.set_xlabel('Choice')
        ax.set_ylabel('Average Length')
    
    # 이미지 크기 분포
    if 'image_stats' in analysis and 'size_stats' in analysis['image_stats']:
        ax = axes[1, 1]
        size_stats = analysis['image_stats']['size_stats']
        ax.text(0.5, 0.5, f"Avg Size: {size_stats['width']['mean']:.0f}x{size_stats['height']['mean']:.0f}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Image Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 데이터 유틸리티 테스트
    print("Testing data utilities...")
    
    # 더미 데이터 생성
    dummy_data = {
        'ID': [f'TEST_{i:03d}' for i in range(100)],
        'img_path': [f'TEST_{i:03d}.jpg' for i in range(100)],
        'Question': [f'What is this? Question {i}' for i in range(100)],
        'A': [f'Answer A {i}' for i in range(100)],
        'B': [f'Answer B {i}' for i in range(100)],
        'C': [f'Answer C {i}' for i in range(100)],
        'D': [f'Answer D {i}' for i in range(100)],
        'answer': np.random.choice(['A', 'B', 'C', 'D'], 100)
    }
    
    df = pd.DataFrame(dummy_data)
    
    # 분석 실행
    analysis = analyze_dataset('dummy_data.csv' if df.to_csv('dummy_data.csv', index=False) is None else 'dummy_data.csv')
    print("Dataset analysis:", analysis)
    
    # 클래스 가중치 계산
    weights = calculate_class_weights(df)
    print("Class weights:", weights)
    
    # 제출 파일 템플릿 생성
    create_submission_template('dummy_data.csv')
    
    print("Data utilities test completed!") 