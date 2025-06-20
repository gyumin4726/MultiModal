import pandas as pd
import os
from typing import List, Dict


def create_baseline_submission(
    predictions: List[str], 
    ids: List[str], 
    sample_submission_path: str = './sample_submission.csv',
    output_path: str = './baseline_submit.csv'
) -> pd.DataFrame:
    """베이스라인 제출 파일 생성
    
    사용자가 제공한 방식대로 sample_submission.csv를 기반으로 제출 파일을 생성합니다.
    
    Args:
        predictions (List[str]): 예측 결과 리스트 ['A', 'B', 'C', 'D']
        ids (List[str]): 샘플 ID 리스트
        sample_submission_path (str): sample_submission.csv 파일 경로
        output_path (str): 출력 파일 경로
        
    Returns:
        pd.DataFrame: 생성된 제출 데이터프레임
    """
    # sample_submission.csv 로드
    if not os.path.exists(sample_submission_path):
        raise FileNotFoundError(f"Sample submission file not found: {sample_submission_path}")
    
    submission = pd.read_csv(sample_submission_path)
    print(f"Loaded sample submission from: {sample_submission_path}")
    print(f"Sample submission shape: {submission.shape}")
    
    # 예측 결과를 딕셔너리로 변환
    id_to_prediction = dict(zip(ids, predictions))
    
    # 결과 할당
    results = []
    for _, row in submission.iterrows():
        sample_id = row['ID']
        if sample_id in id_to_prediction:
            results.append(id_to_prediction[sample_id])
        else:
            # 예측이 없는 경우 기본값 'A' 사용
            results.append('A')
            print(f"Warning: No prediction for ID {sample_id}, using default 'A'")
    
    # 사용자가 제공한 방식대로 결과 할당
    submission['answer'] = results
    
    # 파일 저장
    submission.to_csv(output_path, index=False)
    print(f"✅ Done. Submission saved to: {output_path}")
    
    # 통계 출력
    print(f"Total samples: {len(submission)}")
    print(f"Answer distribution:")
    answer_counts = submission['answer'].value_counts().sort_index()
    for answer, count in answer_counts.items():
        percentage = count / len(submission) * 100
        print(f"  {answer}: {count} ({percentage:.1f}%)")
    
    return submission


def validate_submission_format(submission_path: str) -> bool:
    """제출 파일 형식 검증
    
    Args:
        submission_path (str): 제출 파일 경로
        
    Returns:
        bool: 유효성 여부
    """
    try:
        df = pd.read_csv(submission_path)
        
        # 필수 컬럼 확인
        required_columns = ['ID', 'answer']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ Missing required columns. Expected: {required_columns}, Found: {list(df.columns)}")
            return False
        
        # 답변 형식 확인
        valid_answers = {'A', 'B', 'C', 'D'}
        invalid_answers = set(df['answer']) - valid_answers
        if invalid_answers:
            print(f"❌ Invalid answers found: {invalid_answers}. Only A, B, C, D are allowed.")
            return False
        
        # 중복 ID 확인
        if df['ID'].duplicated().any():
            duplicates = df[df['ID'].duplicated()]['ID'].tolist()
            print(f"❌ Duplicate IDs found: {duplicates}")
            return False
        
        print("✅ Submission format is valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error validating submission: {e}")
        return False


def compare_submissions(file1: str, file2: str) -> Dict:
    """두 제출 파일 비교
    
    Args:
        file1 (str): 첫 번째 제출 파일
        file2 (str): 두 번째 제출 파일
        
    Returns:
        Dict: 비교 결과
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # ID 기준으로 정렬
    df1 = df1.sort_values('ID').reset_index(drop=True)
    df2 = df2.sort_values('ID').reset_index(drop=True)
    
    # 일치하는 예측 수 계산
    matches = (df1['answer'] == df2['answer']).sum()
    total = len(df1)
    agreement = matches / total * 100
    
    # 차이점 분석
    differences = df1[df1['answer'] != df2['answer']][['ID', 'answer']].copy()
    differences['answer_file2'] = df2[df1['answer'] != df2['answer']]['answer'].values
    differences.columns = ['ID', f'answer_{os.path.basename(file1)}', f'answer_{os.path.basename(file2)}']
    
    result = {
        'total_samples': total,
        'matches': matches,
        'agreement_percentage': agreement,
        'differences': differences
    }
    
    print(f"Submission comparison:")
    print(f"  Total samples: {total}")
    print(f"  Matches: {matches}")
    print(f"  Agreement: {agreement:.1f}%")
    print(f"  Differences: {total - matches}")
    
    return result


# 사용 예시
def example_usage():
    """사용 예시"""
    # 더미 예측 결과
    dummy_predictions = ['A', 'B', 'C', 'D'] * 15  # 60개 샘플
    dummy_ids = [f'TEST_{i:03d}' for i in range(60)]
    
    # 더미 sample_submission.csv 생성
    sample_submission = pd.DataFrame({
        'ID': dummy_ids,
        'answer': 'A'  # 기본값
    })
    sample_submission.to_csv('./sample_submission.csv', index=False)
    
    # 제출 파일 생성 (사용자 방식)
    submission = create_baseline_submission(
        predictions=dummy_predictions,
        ids=dummy_ids,
        sample_submission_path='./sample_submission.csv',
        output_path='./baseline_submit.csv'
    )
    
    # 검증
    validate_submission_format('./baseline_submit.csv')
    
    return submission


if __name__ == "__main__":
    # 예시 실행
    example_usage() 