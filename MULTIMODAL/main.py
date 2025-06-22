import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import re
from config import seed_everything
import os
from datetime import datetime

# 모듈 임포트 (안전한 방식으로)
try:
    from vision_encoder import load_vision_encoder
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vision encoder not available - {e}")
    VISION_AVAILABLE = False

try:
    from text_encoder import load_text_encoder
    TEXT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Text encoder not available - {e}")
    TEXT_AVAILABLE = False

try:
    from language_model import load_language_model
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Language model not available - {e}")
    LLM_AVAILABLE = False

try:
    from multimodal_fusion import MultiModalFusion
    FUSION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Multimodal fusion not available - {e}")
    FUSION_AVAILABLE = False

def extract_answer_letter(text):
    """LLM 응답에서 A, B, C, D 답변 추출 (개선된 버전)"""
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
    """모든 모델 로딩 (안전한 방식)"""
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
    
    # Vision Encoder (선택적)
    if VISION_AVAILABLE:
        print("🖼️ Loading Vision Encoder...")
        try:
            vision_encoder = load_vision_encoder(
                model_name='vmamba_tiny_s1l8',
                pretrained_path='./vssm1_tiny_0230s_ckpt_epoch_264.pth',
                output_dim=768,
                frozen_stages=1
            ).to(device)
            models['vision_encoder'] = vision_encoder
            vision_params = count_parameters(vision_encoder)
            total_params += vision_params
            print(f"✅ Vision Encoder loaded! Parameters: {format_parameter_count(vision_params)}")
        except Exception as e:
            print(f"Warning: VMamba not available - {e}")
            models['vision_encoder'] = None
    
    # Text Encoder (선택적)
    if TEXT_AVAILABLE:
        print("📝 Loading Text Encoder...")
        try:
            text_encoder = load_text_encoder(
                model_type='default',
                output_dim=768,
                device=device
            )
            models['text_encoder'] = text_encoder
            text_params = count_parameters(text_encoder)
            total_params += text_params
            print(f"✅ Text Encoder loaded! Parameters: {format_parameter_count(text_params)}")
        except Exception as e:
            print(f"Warning: Text Encoder failed - {e}")
            models['text_encoder'] = None
    
    # Language Model (필수)
    if LLM_AVAILABLE:
        print("🤖 Loading Language Model...")
        try:
            # 빠른 테스트를 위한 모델 선택 (환경변수로 제어 가능)
            import os
            model_choice = os.getenv('LLM_MODEL', 'microsoft/phi-2')  # 기본값: phi-2
            
            language_model = load_language_model(
                model_name=model_choice,
                device=device
            )
            models['language_model'] = language_model
            llm_params = count_parameters(language_model)
            total_params += llm_params
            print(f"✅ Language Model loaded! Parameters: {format_parameter_count(llm_params)}")
        except Exception as e:
            print(f"Warning: Language Model failed - {e}")
            print("    Falling back to text-only mode...")
            models['language_model'] = None
    
    # MultiModal Fusion (선택적)
    if FUSION_AVAILABLE and models['vision_encoder'] is not None and models['text_encoder'] is not None:
        print("🔗 Loading MultiModal Fusion...")
        try:
            multimodal_fusion = MultiModalFusion(
                vision_dim=768,
                text_dim=768,
                hidden_dim=512,
                output_dim=768
            ).to(device)
            models['multimodal_fusion'] = multimodal_fusion
            fusion_params = count_parameters(multimodal_fusion)
            total_params += fusion_params
            print(f"✅ MultiModal Fusion loaded! Parameters: {format_parameter_count(fusion_params)}")
        except Exception as e:
            print(f"Warning: MultiModal Fusion failed - {e}")
            models['multimodal_fusion'] = None
    
    # Image Transform (항상 필요)
    models['transform'] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 로딩된 모델 요약
    print("\n📋 Model Loading Summary:")
    print(f"   Vision Encoder: {'✅' if models['vision_encoder'] else '❌'}")
    print(f"   Text Encoder: {'✅' if models['text_encoder'] else '❌'}")
    print(f"   Language Model: {'✅' if models['language_model'] else '❌'}")
    print(f"   Multimodal Fusion: {'✅' if models['multimodal_fusion'] else '❌'}")
    
    # 총 파라미터 개수 출력
    print("\n🔢 Total Parameter Count:")
    print(f"   Combined Models: {format_parameter_count(total_params)} parameters")
    print(f"   Exact Count: {total_params:,} parameters")
    
    return models

def process_vqa_sample(models, image_path, question, choices, sample_idx):
    """단일 VQA 샘플 처리 (안전한 방식)"""
    try:
        # 이미지 경로 확인
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return 'A'
        
        # Vision Features 추출 (선택적)
        vision_features = None
        if models['vision_encoder'] is not None:
            with torch.no_grad():
                try:
                    vision_features = models['vision_encoder'](image_path)
                except Exception as e:
                    print(f"Warning: Vision encoding failed - {e}")
        
        # 프롬프트는 이제 LLM 내부에서 피처와 함께 구성됨
        
        # Text Features 추출 (선택적)
        text_features = None
        if models['text_encoder'] is not None:
            with torch.no_grad():
                try:
                    text_features = models['text_encoder'](question)
                except Exception as e:
                    print(f"Warning: Text encoding failed - {e}")
        
        # MultiModal Fusion (선택적)
        fused_features = None
        if (models['multimodal_fusion'] is not None and 
            vision_features is not None and 
            text_features is not None):
            with torch.no_grad():
                try:
                    fused_features = models['multimodal_fusion'](vision_features, text_features)
                except Exception as e:
                    print(f"Warning: Multimodal fusion failed - {e}")
        
        # 피처 추출 상태 확인 (디버깅용)
        features_available = []
        if vision_features is not None:
            features_available.append(f"Vision({vision_features.shape})")
        if text_features is not None:
            features_available.append(f"Text({text_features.shape})")
        if fused_features is not None:
            features_available.append(f"Fused({fused_features.shape})")
        
        # LLM 추론 (피처 활용)
        if models['language_model'] is not None:
            try:
                # 피처 기반 추론 시도
                if vision_features is not None or text_features is not None or fused_features is not None:
                    print(f"🚀 Using features: {', '.join(features_available)}")
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
                        print("-" * 50)
                else:
                    print("⚠️ No features available, using text-only mode")
                    # 피처가 없으면 기본 방식
                    response = models['language_model'].answer_question_simple(question, choices)
                    answer = extract_answer_letter(response)
            except Exception as e:
                print(f"Warning: LLM generation failed - {e}")
                answer = 'A'
        else:
            # LLM이 없으면 단순 휴리스틱
            answer = 'A'  # 기본값
            
        return answer
        
    except Exception as e:
        print(f"❌ Error processing sample: {e}")
        return 'A'

def main():
    """메인 추론 함수"""
    # 시드 고정
    seed_everything()
    
    print("🚀 Starting MultiModal VQA Inference...")
    print("="*60)
    
    # 모델 로딩
    models = load_models()
    
    # 최소한 하나의 모델이라도 로딩되었는지 확인
    if all(v is None for k, v in models.items() if k not in ['transform', 'device']):
        print("❌ No models loaded successfully. Exiting...")
        return
    
    # 데이터 로딩
    print("\n📊 Loading test data...")
    if not os.path.exists('./dev_test.csv'):
        print("❌ dev_test.csv not found!")
        return
        
    test = pd.read_csv('./dev_test.csv')
    print(f"📋 Total samples: {len(test)}")
    
    # 추론
    print("\n🔍 Starting inference...")
    results = []
    
    for idx, (_, row) in enumerate(tqdm(test.iterrows(), total=len(test), desc="Processing")):
        image_path = row['img_path']
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
    output_filename = f'baseline_submit_{current_time}.csv'
    
    if os.path.exists('./sample_submission.csv'):
        submission = pd.read_csv('./sample_submission.csv')
        submission['answer'] = results
        submission.to_csv(f'./{output_filename}', index=False)
        print(f"✅ Results saved to '{output_filename}'")
    else:
        # sample_submission.csv가 없으면 직접 생성
        submission = pd.DataFrame({
            'ID': range(len(results)),
            'answer': results
        })
        submission.to_csv(f'./{output_filename}', index=False)
        print(f"✅ Results saved to '{output_filename}' (created new format)")
    
    # 결과 분석
    print(f"\n📊 Answer distribution:")
    for answer in ['A', 'B', 'C', 'D']:
        count = results.count(answer)
        print(f"   {answer}: {count} ({count/len(results)*100:.1f}%)")
    
    print("\n🎉 VQA inference completed successfully!")

if __name__ == "__main__":
    main() 