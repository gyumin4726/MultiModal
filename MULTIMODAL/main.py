import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import re
from config import seed_everything
import os
from datetime import datetime

# 모듈 임포트 (BLIP2 스타일 - 단순화)
from model.vision_encoder import load_vision_encoder
from model.language_model import load_language_model
# text_encoder, multimodal_fusion은 더 이상 사용하지 않음

def extract_answer_letter(text):
    """개선된 답변 추출 - 새로운 프롬프트 형식에 맞게"""
    if not text:
        return 'B'
    
    text = text.strip().upper()
    
    # 1. "A." 형식 우선 검색 (프롬프트에서 요청한 형식)
    match = re.search(r'\b([ABCD])\.\s*$', text)
    if match:
        answer = match.group(1)
        print(f"✅ Found exact format: {answer}.")
        return answer
    
    # 2. "A." 형식 (끝이 아닌 곳에서도)
    match = re.search(r'\b([ABCD])\.', text)
    if match:
        answer = match.group(1)
        print(f"✅ Found letter with period: {answer}.")
        return answer
    
    # 3. 단순히 A, B, C, D만 있는 경우
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        answer = match.group(1)
        print(f"✅ Found single letter: {answer}")
        return answer
    
    # 4. 텍스트 시작 부분의 문자
    if len(text) > 0 and text[0] in 'ABCD':
        answer = text[0]
        print(f"✅ Found at start: {answer}")
        return answer
    
    # 5. 실패시 기본값
    print(f"⚠️ No clear answer found in: '{text}'")
    return 'B'

def count_parameters(model):
    """모델의 총 파라미터 개수 계산"""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters())

def format_parameter_count(count):
    """파라미터 개수를 읽기 쉬운 형태로 변환"""
    if count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    else:
        return str(count)

def validate_image_path(image_path):
    """이미지 경로 검증 및 수정"""
    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_path}")
        return None
    
    try:
        # 이미지 로딩 테스트
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            print(f"✅ Image validation OK: {image_path} ({img.size})")
        return image_path
    except Exception as e:
        print(f"❌ Image loading error: {e}")
        return None

def validate_data_format(row):
    """데이터 형식 검증"""
    required_fields = ['Question', 'A', 'B', 'C', 'D']
    for field in required_fields:
        if field not in row or pd.isna(row[field]) or str(row[field]).strip() == '':
            print(f"❌ Missing or empty field: {field}")
            return False
    return True

def load_models():
    """BLIP2 스타일 모델 로딩"""
    print("🚀 Loading BLIP2-style VQA System...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    models = {
        'vision_encoder': None,
        'language_model': None,
        'device': device
    }
    
    total_params = 0
    
    # Vision Encoder (Q-Former)
    print("🖼️ Loading Vision Encoder with Q-Former...")
    vision_encoder = load_vision_encoder(
        model_name='vit_base_patch16_224',  # 메모리 효율성을 위해 Base 사용
        pretrained=True,
        output_dim=768,
        frozen_stages=2,  # 더 많은 레이어 freeze (빠른 추론)
        num_query_tokens=32  # BLIP2 표준
    ).to(device)
    models['vision_encoder'] = vision_encoder
    vision_params = count_parameters(vision_encoder)
    total_params += vision_params
    print(f"✅ Vision Encoder loaded! Parameters: {format_parameter_count(vision_params)}")
    
    # Language Model (OPT)
    print("🤖 Loading Multimodal Language Model (OPT)...")
    language_model = load_language_model(
        model_name='facebook/opt-2.7b',  # BLIP2에서 사용하는 OPT
        device=device
    )
    models['language_model'] = language_model
    llm_params = count_parameters(language_model)
    total_params += llm_params
    print(f"✅ Language Model loaded! Parameters: {format_parameter_count(llm_params)}")
    
    # 총 파라미터 개수 출력
    print("\n🔢 BLIP2-style System Parameter Count:")
    print(f"   Vision Encoder (Q-Former): {format_parameter_count(vision_params)} parameters")
    print(f"   Language Model (OPT): {format_parameter_count(llm_params)} parameters")
    print(f"   Total System: {format_parameter_count(total_params)} parameters")
    print(f"\n📸 Vision Processing: Q-Former with 32 learnable query tokens")
    print(f"🔗 Multimodal Strategy: Direct vision token injection into LLM (BLIP2 style)")
    
    return models

def process_vqa_sample(models, image_path, question, choices, sample_idx):
    """BLIP2 스타일 VQA 처리"""
    
    print(f"🔍 Processing Question {sample_idx + 1:02d}/60...")
    
    # 이미지 경로 검증
    valid_image_path = validate_image_path(image_path)
    if valid_image_path is None:
        print(f"❌ Skipping due to invalid image: {image_path}")
        return 'B'  # 기본값 반환
    
    # 데이터 검증
    if len(choices) != 4:
        print(f"❌ Invalid choices count: {len(choices)}")
        return 'B'
    
    try:
        # 1. Q-Former로 32개 비전 토큰 생성
        with torch.no_grad():
            vision_tokens, attention_weights = models['vision_encoder'].forward_qformer(valid_image_path)
            print(f"✅ Vision tokens generated: {vision_tokens.shape}")
    
        # 2. 멀티모달 LLM이 비전 토큰을 직접 처리
        response = models['language_model'].answer_vqa_multimodal(
            vision_tokens=vision_tokens,
            question=question,
            choices=choices
        )
        answer = extract_answer_letter(response)
        
        # 처음 3개 샘플만 상세 출력
        if sample_idx < 3:
            print(f"📝 Q{sample_idx+1}: {question}")
            print(f"🔍 Choices: {choices}")
            print(f"🤖 LLM Response: '{response}'")
            print(f"🎯 Answer: {answer}")
        else:
            print(f"✅ Q{sample_idx+1:02d} → {answer}")
        
        return answer
        
    except Exception as e:
        print(f"❌ Error processing sample {sample_idx+1}: {e}")
        return 'B'  # 에러 시 기본값

def main():
    """메인 추론 함수"""
    # 시드 고정
    seed_everything()
    
    print("🚀 Starting BLIP2-style VQA Inference...")
    print("="*60)
    
    # 모델 로딩
    models = load_models()
    
    # 데이터 로딩 및 검증
    print("\n📊 Loading and validating test data...")
    
    if not os.path.exists('./data/dev_test.csv'):
        print("❌ Test data file not found: ./data/dev_test.csv")
        return
    
    test = pd.read_csv('./data/dev_test.csv')
    print(f"📋 Total samples: {len(test)}")
    
    # 데이터 형식 검증
    print("🔍 Validating data format...")
    valid_samples = 0
    for idx, row in test.iterrows():
        if validate_data_format(row):
            valid_samples += 1
        else:
            print(f"❌ Invalid data format at row {idx}")
    
    print(f"✅ Valid samples: {valid_samples}/{len(test)}")
    
    # 이미지 디렉토리 확인
    image_dir = './data/input_images/'
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    print(f"✅ Image directory found: {image_dir}")
    available_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    print(f"✅ Available images: {available_images}")
    
    # 정답지 로딩 (선택적)
    ground_truth = None
    try:
        ground_truth_df = pd.read_csv('./data/dev_ans.csv')
        ground_truth = []
        for _, row in ground_truth_df.iterrows():
            answer_col = str(row.iloc[-1]).strip().upper()
            if answer_col in ['A', 'B', 'C', 'D']:
                ground_truth.append(answer_col)
            elif answer_col == 'ABC':
                ground_truth.append('A')
            elif answer_col == '?':
                ground_truth.append(None)
            else:
                ground_truth.append(None)
        print(f"✅ Ground truth loaded: {len([x for x in ground_truth if x is not None])}/{len(ground_truth)} valid answers")
    except FileNotFoundError:
        print("⚠️ Ground truth file (dev_ans.csv) not found - accuracy will not be calculated")
    except Exception as e:
        print(f"⚠️ Error loading ground truth: {e}")
        ground_truth = None
    
    # 추론
    print("\n🔍 Starting inference...")
    results = []
    answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    for idx, (_, row) in enumerate(test.iterrows()):
        # 이미지 경로 생성
        img_filename = os.path.basename(row['img_path'])
        image_path = os.path.join(image_dir, img_filename)
        
        # 데이터 추출
        question = str(row['Question']).strip()
        choices = [str(row[c]).strip() for c in ['A', 'B', 'C', 'D']]
        
        # VQA 처리
        answer = process_vqa_sample(models, image_path, question, choices, sample_idx=idx)
        results.append(answer)
        
        # 답변 선택 카운트
        if answer in answer_counts:
            answer_counts[answer] += 1
    
    print('✅ Inference completed!')
    
    # 결과 저장
    print("\n📁 Saving results...")
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'vqa_blip2_submit_{current_time}.csv'
    
    submission = pd.DataFrame({
        'ID': [f'TEST_{i:03d}' for i in range(len(results))],
        'answer': results
    })
    submission.to_csv(f'./data/{output_filename}', index=False)
    print(f"✅ Results saved to 'data/{output_filename}'")
    
    # 최종 결과 요약
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS SUMMARY")
    print("="*60)
    
    total_questions = len(results)
    print(f"📋 Total Questions: {total_questions}")
    
    # 답변 분포
    print(f"\n📊 Answer Distribution:")
    for option in ['A', 'B', 'C', 'D']:
        count = answer_counts[option]
        percentage = (count / total_questions * 100) if total_questions > 0 else 0
        print(f"   {option}: {count:2d}회 ({percentage:5.1f}%)")
    
    # 답변 패턴 분석
    most_selected = max(answer_counts, key=answer_counts.get)
    least_selected = min(answer_counts, key=answer_counts.get)
    
    print(f"\n🔍 Answer Pattern Analysis:")
    print(f"   Most Selected: {most_selected} ({answer_counts[most_selected]}회)")
    print(f"   Least Selected: {least_selected} ({answer_counts[least_selected]}회)")
    
    # 전체 답변 리스트
    print(f"\n📝 All Answers: {' '.join(results)}")
    
    # 정확도 계산 (정답지가 있는 경우)
    if ground_truth is not None:
        print(f"\n🎯 ACCURACY EVALUATION")
        print("-" * 40)
        
        correct_count = 0
        valid_count = 0
        
        for i, (pred, true) in enumerate(zip(results, ground_truth)):
            if true is not None:
                valid_count += 1
                if pred == true:
                    correct_count += 1
        
        accuracy = (correct_count / valid_count * 100) if valid_count > 0 else 0
        
        print(f"✅ Correct: {correct_count}/{valid_count} questions")
        print(f"🎯 Accuracy: {accuracy:.1f}%")
        
        if accuracy >= 80:
            grade = "🏆 Excellent"
        elif accuracy >= 70:
            grade = "🥇 Good"
        elif accuracy >= 60:
            grade = "🥈 Fair"
        elif accuracy >= 50:
            grade = "🥉 Needs Improvement"
        else:
            grade = "❌ Poor"
        
        print(f"📊 Performance Grade: {grade}")
    
    print("="*60)
    print("\n🎉 BLIP2-style VQA inference completed successfully!")

if __name__ == "__main__":
    main() 