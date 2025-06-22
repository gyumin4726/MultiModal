import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from config import device
from utils import extract_answer_letter
from vision_encoder import load_vision_encoder
from text_encoder import load_text_encoder
from language_model import load_language_model

def setup_models():
    """모든 모델 로딩"""
    print("🔄 Setting up MultiModal models...")
    
    # 비전 인코더 로딩
    print("📸 Loading Vision Encoder (VMamba)...")
    vision_encoder = load_vision_encoder(
        model_name='vmamba_tiny_s1l8',
        pretrained_path='./vssm1_tiny_0230s_ckpt_epoch_264.pth',
        output_dim=768,
        frozen_stages=1
    )
    
    # 텍스트 인코더 로딩
    print("📝 Loading Text Encoder (sentence-transformers)...")
    text_encoder = load_text_encoder(
        model_type='default',
        output_dim=768
    )
    
    # 언어 모델 로딩
    print("🤖 Loading Language Model (microsoft/phi-2)...")
    language_model = load_language_model(model_name="microsoft/phi-2")
    
    # GPU로 이동
    if torch.cuda.is_available():
        vision_encoder = vision_encoder.cuda()
        # text_encoder와 language_model은 이미 내부적으로 GPU 처리됨
    
    print("✅ All models loaded successfully!")
    return vision_encoder, text_encoder, language_model

def setup_image_transform():
    """이미지 전처리 파이프라인 설정"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.675/255, 116.28/255, 103.53/255],
                           std=[58.395/255, 57.12/255, 57.375/255])
    ])

def create_multimodal_prompt(question, choices, vision_features=None, text_features=None):
    """멀티모달 프롬프트 생성"""
    prompt = (
        "You are a helpful AI that answers multiple-choice questions based on images.\n"
        "Analyze the image carefully and select the best answer from the given choices.\n\n"
        f"Question: {question}\n"
        "Choices:\n"
    )
    
    # 선택지 추가
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    prompt += "\nBased on the image, the answer is:"
    
    return prompt

def run_multimodal_inference():
    """멀티모달 추론 실행"""
    # 모델 설정
    vision_encoder, text_encoder, language_model = setup_models()
    transform = setup_image_transform()
    
    # 데이터 로딩
    print("📊 Loading test data...")
    test_data = pd.read_csv('./dev_test.csv')
    print(f"Total samples: {len(test_data)}")
    
    results = []
    
    print("🚀 Starting inference...")
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing"):
        try:
            # 이미지 로딩 및 전처리
            image_path = row['img_path']
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
            
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            # 질문 및 선택지
            question = row['Question']
            choices = [row['A'], row['B'], row['C'], row['D']]
            
            # 1. 비전 인코더로 이미지 피처 추출
            with torch.no_grad():
                vision_features = vision_encoder(image_tensor)
            
            # 2. 텍스트 인코더로 질문 임베딩 (선택사항)
            question_embedding = text_encoder([question])
            
            # 3. 멀티모달 프롬프트 생성
            prompt = create_multimodal_prompt(question, choices)
            
            # 4. LLM으로 답변 생성
            response = language_model.generate_text(
                prompt, 
                max_new_tokens=10,  # 짧은 답변만 필요
                temperature=0.0
            )
            
            # 5. 답변에서 알파벳 추출
            predicted_answer = extract_answer_letter(response)
            
            results.append(predicted_answer)
            
            # 진행 상황 출력 (처음 5개만)
            if idx < 5:
                print(f"\n--- Sample {idx} ---")
                print(f"Image: {image_path}")
                print(f"Question: {question}")
                print(f"Choices: {choices}")
                print(f"Response: {response}")
                print(f"Predicted: {predicted_answer}")
                print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error processing {row['ID']}: {e}")
            results.append("?")  # 오류 시 기본값
    
    print('\n✅ Inference completed!')
    
    # 결과 저장
    submission = pd.read_csv('./sample_submission.csv')
    submission['answer'] = results
    submission.to_csv('./multimodal_submission.csv', index=False)
    print("✅ Results saved to 'multimodal_submission.csv'")
    
    # 간단한 통계
    answer_counts = pd.Series(results).value_counts()
    print(f"\n📊 Answer distribution:")
    print(answer_counts)
    
    return results

if __name__ == "__main__":
    # 멀티모달 추론 실행
    run_multimodal_inference() 