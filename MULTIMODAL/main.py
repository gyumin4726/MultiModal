import torch
from PIL import Image
import torchvision.transforms as transforms
from config import seed_everything
from vision_encoder import load_vision_encoder
from text_encoder import load_text_encoder, list_available_models
from language_model import load_language_model

def test_vision_encoder():
    """비전 인코더 테스트"""
    print("🔍 Loading vision encoder...")
    try:
        vision_encoder = load_vision_encoder(
            model_name='vmamba_tiny_s1l8',
            pretrained_path='./vssm1_tiny_0230s_ckpt_epoch_264.pth',
            output_dim=768,
            frozen_stages=1
        )
        print("✅ Vision encoder loaded successfully!")
        
        # GPU 로딩
        if torch.cuda.is_available():
            vision_encoder = vision_encoder.cuda()
        
        # 이미지 전처리 파이프라인
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675/255, 116.28/255, 103.53/255],
                               std=[58.395/255, 57.12/255, 57.375/255])
        ])
        
        # 테스트 이미지 로딩
        test_image_path = "input_images/TEST_000.jpg"
        try:
            image = Image.open(test_image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
            
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            # 비전 인코더 추론
            with torch.no_grad():
                vision_features = vision_encoder(image_tensor)
                print(f"✅ Vision features shape: {vision_features.shape}")
                print(f"✅ Vision features norm: {vision_features.norm().item():.4f}")
                
        except FileNotFoundError:
            print("⚠️ Test image not found, creating dummy tensor for testing...")
            dummy_image = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_image = dummy_image.cuda()
            
            with torch.no_grad():
                vision_features = vision_encoder(dummy_image)
                print(f"✅ Vision features shape: {vision_features.shape}")
                print(f"✅ Vision features norm: {vision_features.norm().item():.4f}")
        
        return vision_encoder
        
    except Exception as e:
        print(f"❌ Error loading vision encoder: {e}")
        return None

def test_text_encoder():
    """텍스트 인코더 테스트"""
    print("📝 Loading text encoder...")
    try:
        # 사용 가능한 모델 목록 출력
        print("\n📋 Available text encoder models:")
        list_available_models()
        
        # 기본 텍스트 인코더 로딩 (all-MiniLM-L6-v2)
        text_encoder = load_text_encoder(
            model_type='default',
            output_dim=768  # VMamba와 맞춤
        )
        print("✅ Text encoder loaded successfully!")
        
        # 테스트 텍스트들
        test_texts = [
            "What is shown in this image?",
            "Describe the main objects in the picture.",
            "Can you identify the key features in this photo?",
            "What are the colors and shapes visible?"
        ]
        
        # 텍스트 인코딩 테스트
        print(f"🔤 Encoding {len(test_texts)} test sentences...")
        with torch.no_grad():
            text_features = text_encoder(test_texts)
            print(f"✅ Text features shape: {text_features.shape}")
            print(f"✅ Text features norm: {text_features.norm(dim=1).mean().item():.4f}")
            
            # 개별 문장별 결과
            for i, text in enumerate(test_texts):
                norm = text_features[i].norm().item()
                print(f"   📄 Sentence {i+1}: norm={norm:.4f}")
        
        return text_encoder
        
    except Exception as e:
        print(f"❌ Error loading text encoder: {e}")
        print("💡 Make sure sentence-transformers is installed: pip install sentence-transformers")
        return None

def test_language_model():
    """언어 모델 테스트"""
    print("🤖 Loading language model...")
    try:
        language_model = load_language_model(model_name="microsoft/phi-2")
        print("✅ Language model loaded successfully!")
        
        # 텍스트 생성 테스트
        prompt = "Explain what a black hole is in simple terms."
        result = language_model.generate_text(prompt, max_new_tokens=100)
        print(f"Generated text: {result}")
        
        return language_model
        
    except Exception as e:
        print(f"❌ Error loading language model: {e}")
        return None

if __name__ == "__main__":
    # 시드 고정
    seed_everything()
    
    print("🚀 Starting complete MultiModal system test...")
    print("="*60)
    
    # 개별 모듈 테스트
    print("\n1️⃣ Testing VMamba Vision Encoder")
    print("-"*40)
    vision_encoder = test_vision_encoder()
    
    print("\n2️⃣ Testing sentence-transformers Text Encoder")
    print("-"*40)
    text_encoder = test_text_encoder()
    
    print("\n3️⃣ Testing microsoft/phi-2 Language Model")
    print("-"*40)
    language_model = test_language_model()
    
    # 최종 결과
    print("\n" + "="*60)
    print("📊 FINAL RESULTS:")
    print(f"   Vision Encoder (VMamba): {'✅ OK' if vision_encoder else '❌ FAILED'}")
    print(f"   Text Encoder (BERT): {'✅ OK' if text_encoder else '❌ FAILED'}")
    print(f"   Language Model (phi-2): {'✅ OK' if language_model else '❌ FAILED'}")
    
    if all([vision_encoder, text_encoder, language_model]):
        print("\n🎉 All three modules are working perfectly!")
        print("🔥 Ready to implement full multimodal integration!")
        print("\n📋 Current architecture:")
        print("   🖼️ Vision: VMamba (사전학습)")
        print("   📝 Text: sentence-transformers (사전학습)")  
        print("   🤖 LLM: microsoft/phi-2 (사전학습)")
    else:
        print("\n⚠️ Some modules need attention.")
        print("🔧 Please check the error messages above.") 