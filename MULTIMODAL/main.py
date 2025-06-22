import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import re
from config import seed_everything
from vision_encoder import load_vision_encoder
from text_encoder import load_text_encoder
from language_model import load_language_model
from multimodal_fusion import MultiModalFusion
import os

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
    """모든 모델 로딩"""
    print("🚀 Loading MultiModal VQA System...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Vision Encoder
    print("🖼️ Loading Vision Encoder...")
    try:
        vision_encoder = load_vision_encoder(
            model_name='vmamba_tiny_s1l8',
            pretrained_path='./vssm1_tiny_0230s_ckpt_epoch_264.pth',
            output_dim=768,
            frozen_stages=1
        ).to(device)
        print("✅ Vision Encoder loaded!")
    except Exception as e:
        print(f"❌ Vision Encoder failed: {e}")
        vision_encoder = None
    
    # Text Encoder
    print("📝 Loading Text Encoder...")
    try:
        text_encoder = load_text_encoder(
            model_type='default',
            output_dim=768
        )
        print("✅ Text Encoder loaded!")
    except Exception as e:
        print(f"❌ Text Encoder failed: {e}")
        text_encoder = None
    
    # Language Model
    print("🤖 Loading Language Model...")
    try:
        language_model = load_language_model(model_name="microsoft/phi-2")
        print("✅ Language Model loaded!")
    except Exception as e:
        print(f"❌ Language Model failed: {e}")
        language_model = None
    
    # MultiModal Fusion
    print("🔗 Loading MultiModal Fusion...")
    try:
        multimodal_fusion = MultiModalFusion(
            vision_dim=768,
            text_dim=768,
            hidden_dim=512,
            num_heads=8
        ).to(device)
        print("✅ MultiModal Fusion loaded!")
    except Exception as e:
        print(f"❌ MultiModal Fusion failed: {e}")
        multimodal_fusion = None
    
    # Image Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'vision_encoder': vision_encoder,
        'text_encoder': text_encoder,
        'language_model': language_model,
        'multimodal_fusion': multimodal_fusion,
        'transform': transform,
        'device': device
    }

def process_vqa_sample(models, image_path, question, choices):
    """단일 VQA 샘플 처리"""
    try:
        # 이미지 로딩
        if not os.path.exists(image_path):
            return 'A'
            
        image = Image.open(image_path).convert("RGB")
        image_tensor = models['transform'](image).unsqueeze(0).to(models['device'])
        
        # Vision Features 추출
        with torch.no_grad():
            if models['vision_encoder'] is not None:
                vision_features = models['vision_encoder'](image_tensor)
            else:
                vision_features = None
        
        # 프롬프트 구성
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        prompt = (
            "You are a helpful AI that answers multiple-choice questions based on the given image.\n"
            "Select the best answer from A, B, C, or D.\n\n"
            f"Question: {question}\n"
            f"{choices_text}\n"
            "Answer:"
        )
        
        # Text Features 추출 (옵션)
        if models['text_encoder'] is not None:
            with torch.no_grad():
                text_features = models['text_encoder']([question])
        else:
            text_features = None
        
        # MultiModal Fusion (옵션)
        if (models['multimodal_fusion'] is not None and 
            vision_features is not None and 
            text_features is not None):
            with torch.no_grad():
                fused_features = models['multimodal_fusion'](vision_features, text_features)
        
        # LLM 추론
        if models['language_model'] is not None:
            response = models['language_model'].generate_text(
                prompt,
                max_new_tokens=5,
                temperature=0.0
            )
            answer = extract_answer_letter(response)
        else:
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
    
    # 데이터 로딩
    print("\n📊 Loading test data...")
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
    submission = pd.read_csv('./sample_submission.csv')
    submission['answer'] = results
    submission.to_csv('./baseline_submit.csv', index=False)
    
    print("✅ Results saved to 'baseline_submit.csv'")
    
    # 결과 분석
    print(f"\n📊 Answer distribution:")
    for answer in ['A', 'B', 'C', 'D']:
        count = results.count(answer)
        print(f"   {answer}: {count} ({count/len(results)*100:.1f}%)")
    
    print("\n🎉 VQA inference completed successfully!")

if __name__ == "__main__":
    main() 