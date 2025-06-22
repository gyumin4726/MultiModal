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
    """응답에서 A, B, C, D 추출"""
    text = text.strip().upper()
    
    # 패턴 매칭으로 답변 추출
    patterns = [
        r'\b([ABCD])\b',  # 단독 문자
        r'답:\s*([ABCD])',  # 답: A 형태
        r'정답:\s*([ABCD])',  # 정답: A 형태
        r'Answer:\s*([ABCD])',  # Answer: A 형태
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # 첫 번째로 나타나는 A, B, C, D 반환
    for char in text:
        if char in 'ABCD':
            return char
            
    return 'A'  # 기본값

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
            print("✅ Vision Encoder loaded!")
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
            print("✅ Text Encoder loaded!")
        except Exception as e:
            print(f"Warning: Text Encoder failed - {e}")
            models['text_encoder'] = None
    
    # Language Model (필수)
    if LLM_AVAILABLE:
        print("🤖 Loading Language Model...")
        try:
            language_model = load_language_model(
                model_name="microsoft/phi-2",
                device=device
            )
            models['language_model'] = language_model
            print("✅ Language Model loaded!")
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
                num_heads=8
            ).to(device)
            models['multimodal_fusion'] = multimodal_fusion
            print("✅ MultiModal Fusion loaded!")
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
    
    return models

def process_vqa_sample(models, image_path, question, choices):
    """단일 VQA 샘플 처리 (안전한 방식)"""
    try:
        # 이미지 로딩
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return 'A'
            
        image = Image.open(image_path).convert("RGB")
        image_tensor = models['transform'](image).unsqueeze(0).to(models['device'])
        
        # Vision Features 추출 (선택적)
        vision_features = None
        if models['vision_encoder'] is not None:
            with torch.no_grad():
                try:
                    vision_features = models['vision_encoder'](image_tensor)
                except Exception as e:
                    print(f"Warning: Vision encoding failed - {e}")
        
        # 프롬프트 구성
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        prompt = (
            "You are a helpful AI that answers multiple-choice questions.\n"
            "Select the best answer from A, B, C, or D.\n\n"
            f"Question: {question}\n"
            f"{choices_text}\n\n"
            "Answer:"
        )
        
        # Text Features 추출 (선택적)
        text_features = None
        if models['text_encoder'] is not None:
            with torch.no_grad():
                try:
                    text_features = models['text_encoder']([question])
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
        
        # LLM 추론
        if models['language_model'] is not None:
            try:
                response = models['language_model'].generate_text(
                    prompt,
                    max_new_tokens=10,
                    temperature=0.0
                )
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
    
    for _, row in tqdm(test.iterrows(), total=len(test), desc="Processing"):
        image_path = row['img_path']
        question = row['Question']
        choices = [row[c] for c in ['A', 'B', 'C', 'D']]
        
        # VQA 처리
        answer = process_vqa_sample(models, image_path, question, choices)
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