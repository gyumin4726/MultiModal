import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import re
from config import seed_everything
import os
from datetime import datetime

# 모듈 임포트
from model.vision_encoder import load_vision_encoder
from model.text_encoder import load_vqa_text_encoder
from model.language_model import load_language_model
from model.multimodal_fusion import HierarchicalVQAFusion

def extract_answer_letter(text):
    """LLM 응답에서 A, B, C, D 답변 추출"""
    if not text:
        print(f"⚠️ Empty response, defaulting to A")
        return 'A'
    
    text = text.strip().upper()
    
    # 다양한 패턴 검색 (우선순위 순)
    patterns = [
        r'\b([ABCD])\b(?:\s*[.:]|\s*$)',  # 단독 문자 + 마침표/콜론/끝
        r'ANSWER[:\s]*([ABCD])',           # "ANSWER: A" 형태
        r'SOLUTION[:\s]*([ABCD])',         # "SOLUTION: A" 형태  
        r'([ABCD])[.:]',                   # "A." 또는 "A:" 형태
        r'\b([ABCD])\b',                   # 단순 문자 매칭 (마지막 수단)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1]  # 마지막 매치 사용
            print(f"✅ Extracted answer '{answer}' using pattern: {pattern}")
            return answer
    
    # 모든 패턴 실패시 디버깅 정보 출력
    print(f"❌ No answer pattern found in: '{text[:100]}...'")
    print(f"⚠️ Defaulting to A")
    return 'A'

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

def load_models():
    """모든 모델 로딩"""
    print("🚀 Loading MultiModal VQA System...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    models = {
        'vision_encoder': None,
        'text_encoder': None,
        'language_model': None,
        'multimodal_fusion': None,
        'device': device
    }
    
    total_params = 0
    
    # Vision Encoder
    print("🖼️ Loading Vision Encoder...")
    vision_encoder = load_vision_encoder(
        model_name='vit_large_patch16_224',  # 최고 성능 ViT 모델
        pretrained=True,  # ImageNet 사전 훈련 가중치 사용
        output_dim=1024,
        frozen_stages=1,
        use_skip_connection=True  # MASC-V 활성화
    ).to(device)
    models['vision_encoder'] = vision_encoder
    vision_params = count_parameters(vision_encoder)
    total_params += vision_params
    print(f"✅ Vision Encoder loaded! Parameters: {format_parameter_count(vision_params)}")
    
    # VQA Text Encoder
    print("📝 Loading VQA-Optimized Text Encoder...")
    text_encoder = load_vqa_text_encoder(
        model_type='vqa_optimized',
        output_dim=1024,
        device=device
    )
    models['text_encoder'] = text_encoder
    text_params = count_parameters(text_encoder)
    total_params += text_params
    print(f"✅ VQA Text Encoder loaded! Parameters: {format_parameter_count(text_params)}")
    
    # Language Model
    print("🤖 Loading Language Model...")
    language_model = load_language_model(
        model_name='microsoft/phi-2',
        device=device
    )
    models['language_model'] = language_model
    llm_params = count_parameters(language_model)
    total_params += llm_params
    print(f"✅ Language Model loaded! Parameters: {format_parameter_count(llm_params)}")
    
    # VQA MultiModal Fusion
    print("🔗 Loading VQA Hierarchical Fusion...")
    multimodal_fusion = HierarchicalVQAFusion(
        vision_dim=1024,
        text_dim=1024,
        output_dim=1024
    ).to(device)
    models['multimodal_fusion'] = multimodal_fusion
    fusion_params = count_parameters(multimodal_fusion)
    total_params += fusion_params
    print(f"✅ VQA Hierarchical Fusion loaded! Parameters: {format_parameter_count(fusion_params)}")
    
    # Image Transform
    models['transform'] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 총 파라미터 개수 출력
    print("\n🔢 Total Parameter Count:")
    print(f"   Combined Models: {format_parameter_count(total_params)} parameters")
    print(f"   Exact Count: {total_params:,} parameters")
    
    return models

def process_vqa_sample(models, image_path, question, choices, sample_idx):
    """단일 VQA 샘플 처리"""
    
    # 문제 번호 및 구분선 출력
    print(f"\n{'='*60}")
    print(f"📝 Question {sample_idx + 1:02d}/60 - Processing: {image_path}")
    print(f"{'='*60}")
    
    # Vision Features 추출
    with torch.no_grad():
        vision_features = models['vision_encoder'](image_path)
    
    # VQA Text Features 추출 (Question + Choices 구조화)
    with torch.no_grad():
        text_features, qc_attention = models['text_encoder'](question, choices)
    
    # MultiModal Fusion
    with torch.no_grad():
        fused_features = models['multimodal_fusion'](vision_features, text_features)
    
    # 피처 정보 출력 (디버깅용)
    features_info = [
        f"Vision({vision_features.shape})",
        f"Text({text_features.shape})",
        f"Fused({fused_features.shape})"
    ]
    
    # LLM 추론 (피처 활용)
    print(f"🚀 Using features: {', '.join(features_info)}")
    response = models['language_model'].answer_question_with_features(
        question=question,
        choices=choices,
        vision_features=vision_features,
        text_features=text_features,
        fused_features=fused_features
    )
    answer = extract_answer_letter(response)
    
    # 디버깅: 처음 3개 샘플의 응답 출력
    if sample_idx < 3:
        print(f"🔍 Sample {sample_idx+1} LLM Response: '{response}'")
        print(f"🔍 Response length: {len(response)}")
        print(f"🎯 Extracted Answer: {answer}")
    
    print(f"✅ Question {sample_idx + 1:02d} completed → Answer: {answer}")
    print(f"{'='*60}\n")
            
    return answer

def main():
    """메인 추론 함수"""
    # 시드 고정
    seed_everything()
    
    print("🚀 Starting MultiModal VQA Inference...")
    print("="*60)
    
    # 모델 로딩
    models = load_models()
    
    # 데이터 로딩
    print("\n📊 Loading test data...")
    test = pd.read_csv('./data/dev_test.csv')
    print(f"📋 Total samples: {len(test)}")
    
    # 추론
    print("\n🔍 Starting inference...")
    results = []
    
    # tqdm을 disable하여 progress bar 대신 우리만의 출력 사용
    for idx, (_, row) in enumerate(test.iterrows()):
        # 이미지 경로 수정: 보다 안전한 경로 처리
        img_filename = os.path.basename(row['img_path'])  # TEST_000.jpg 추출 
        image_path = os.path.join('./data/input_images/', img_filename)
        question = row['Question']
        choices = [row[c] for c in ['A', 'B', 'C', 'D']]
        
        # VQA 처리
        answer = process_vqa_sample(models, image_path, question, choices, sample_idx=idx)
        results.append(answer)
    
    print('✅ Inference completed!')
    
    # 결과 저장
    print("\n📁 Saving results...")
    
    # 현재 날짜와 시간으로 파일명 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'vqa_enhanced_submit_{current_time}.csv'
    
    submission = pd.DataFrame({
        'ID': [f'TEST_{i:03d}' for i in range(len(results))],
        'answer': results
    })
    submission.to_csv(f'./data/{output_filename}', index=False)
    print(f"✅ Results saved to 'data/{output_filename}'")
    
    # 결과 분석
    print(f"\n📊 Answer distribution:")
    for answer in ['A', 'B', 'C', 'D']:
        count = results.count(answer)
        print(f"   {answer}: {count} ({count/len(results)*100:.1f}%)")
    
    print("\n🎉 Enhanced VQA inference completed successfully!")

if __name__ == "__main__":
    main() 