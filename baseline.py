import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

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

# CLIP 모델 로드
clip_model_name = "openai/clip-vit-base-patch16"
clip = CLIPModel.from_pretrained(clip_model_name).to(device)
processor = CLIPProcessor.from_pretrained(clip_model_name)
clip.eval()

# 예측 함수
def predict(image_path, question, choices):
    image = Image.open(image_path).convert("RGB")
    prompts = [f"{question} 선택지: {choice}" for choice in choices]
    inputs = processor(text=prompts, images=[image]*4, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip(**inputs)
        logits = outputs.logits_per_image  # (4,)
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
submission.to_csv('./baseline_submit.csv', index=False)
print("✅ Done. Saved to ./baseline_submit.csv")

# 정확도 계산
ground_truth = pd.read_csv('./dev_ans.csv')
predictions = pd.read_csv('./baseline_submit.csv')

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
