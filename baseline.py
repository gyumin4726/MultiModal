import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from ss2d import SS2D

# 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# Seed 고정
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

class EnhancedCLIP(torch.nn.Module):
    def __init__(self, clip_model_name):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # SS2D 레이어 추가 - CLIP ViT-H-14의 hidden size는 1024입니다
        self.ss2d = SS2D(
            d_model=1024,
            d_state=32,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
        )
        
    def forward(self, image_embeds, text_embeds):
        # 이미지 임베딩에 SS2D 적용 (B, L, D) -> (B, L, D)
        enhanced_image_embeds, _ = self.ss2d(image_embeds)
        
        # 정규화된 특징 계산
        image_embeds = enhanced_image_embeds / enhanced_image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # 코사인 유사도 계산
        logits = torch.matmul(image_embeds, text_embeds.transpose(-2, -1))
        
        return logits

# CLIP 모델 로드 - LAION의 ViT-Huge 모델 사용
clip_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
print(f"✅ Loading model: {clip_model_name}")
model = EnhancedCLIP(clip_model_name).to(device)
processor = CLIPProcessor.from_pretrained(clip_model_name)
model.eval()

# 예측 함수
def predict(image_path, question, choices):
    image = Image.open(image_path).convert("RGB")
    prompts = [f"{question} 선택지: {choice}" for choice in choices]
    inputs = processor(text=prompts, images=[image]*4, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        # CLIP의 이미지/텍스트 임베딩 추출
        image_features = model.clip.vision_model(inputs.pixel_values)[0]  # (B, L, D)
        text_features = model.clip.text_model(input_ids=inputs.input_ids, 
                                            attention_mask=inputs.attention_mask)[0]  # (B, L, D)
        
        # SS2D를 포함한 enhanced 모델로 예측
        logits = model(image_features, text_features)  # (B, B)
        logits = logits.mean(dim=0)  # (B,)
        probs = logits.softmax(dim=-1)
        pred = probs.argmax().item()

    return pred

# 데이터 로딩
test = pd.read_csv('./dev_test.csv')  # img_path, Question, A, B, C, D
submission = pd.read_csv('./sample_submission.csv')  # id, answer

results = []

# 추론
for _, row in tqdm(test.iterrows(), total=len(test)):
    image_path = row['img_path']
    question = row['Question']
    choices = [row['A'], row['B'], row['C'], row['D']]

    pred = predict(image_path, question, choices)
    results.append(chr(65 + pred))

# 결과 저장
submission['answer'] = results
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'./baseline_submit_{current_time}.csv'
submission.to_csv(output_filename, index=False)
print(f"✅ Done. Saved to {output_filename}")

# 정확도 계산
ground_truth = pd.read_csv('./dev_ans.csv')
predictions = pd.read_csv(output_filename)

# dev_ans.csv의 마지막 컬럼이 정답이므로, 마지막 컬럼을 선택
ground_truth_answers = ground_truth.iloc[:, -1]

# 유효한 답안만 선택 (?, ABC 등의 특수 답안 제외)
valid_mask = ground_truth_answers.apply(lambda x: len(str(x)) == 1 and str(x).isalpha())
valid_ground_truth = ground_truth_answers[valid_mask]
valid_predictions = predictions['answer'][valid_mask]

# 정확도 계산
correct = (valid_ground_truth == valid_predictions).sum()
total = len(valid_ground_truth)
accuracy = correct / total * 100

print(f"\n✅ 정확도 계산 결과:")
print(f"전체 문제 수: {len(ground_truth)}개")
print(f"유효한 문제 수: {total}개")
print(f"정답 개수: {correct}개")
print(f"정확도: {accuracy:.2f}%")

# 틀린 문제 분석
print("\n❌ 틀린 문제 분석:")
print("-" * 100)

# test 데이터에서 유효한 문제만 선택
valid_test = test[valid_mask].copy()
valid_test['Predicted'] = valid_predictions
valid_test['Correct'] = valid_ground_truth
valid_test['Is_Correct'] = valid_test['Predicted'] == valid_test['Correct']

# 틀린 문제만 선택
wrong_predictions = valid_test[~valid_test['Is_Correct']]

for idx, row in wrong_predictions.iterrows():
    print(f"\n문제: {row['Question']}")
    print(f"이미지 경로: {row['img_path']}")
    print(f"선택지:")
    print(f"A: {row['A']}")
    print(f"B: {row['B']}")
    print(f"C: {row['C']}")
    print(f"D: {row['D']}")
    print(f"예측: {row['Predicted']}")
    print(f"정답: {row['Correct']}")
    print("-" * 100)
