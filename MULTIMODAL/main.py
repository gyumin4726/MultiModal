from config import seed_everything
from model import load_model
from inference import run_inference

if __name__ == "__main__":
    # 시드 고정
    seed_everything()
    
    # 모델 로딩
    processor, model = load_model()
    
    # 추론 실행
    results = run_inference(processor, model) 