import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from collections import Counter
import cv2

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

# 색상 정의 및 매칭
COLOR_MAPPINGS = {
    'red': ([0, 50, 50], [10, 255, 255]),  # 빨강
    'green': ([35, 50, 50], [85, 255, 255]),  # 초록
    'blue': ([100, 50, 50], [130, 255, 255]),  # 파랑
    'yellow': ([20, 50, 50], [35, 255, 255]),  # 노랑
    'black': ([0, 0, 0], [180, 255, 30]),  # 검정
    'white': ([0, 0, 200], [180, 30, 255]),  # 흰색
    'brown': ([10, 50, 20], [20, 255, 200]),  # 갈색
    'gray': ([0, 0, 40], [180, 30, 220]),  # 회색
    'pink': ([150, 50, 50], [170, 255, 255]),  # 분홍
}

def get_color_distribution(image, region=None):
    # 이미지를 HSV 색공간으로 변환
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    
    if region == 'sky':
        # 하늘은 이미지 상단 1/3 영역으로 가정
        height = img_cv.shape[0]
        img_cv = img_cv[:height//3, :, :]
    elif region == 'ground':
        # 지면은 이미지 하단 1/3 영역으로 가정
        height = img_cv.shape[0]
        img_cv = img_cv[2*height//3:, :, :]
    
    color_counts = Counter()
    total_pixels = img_cv.shape[0] * img_cv.shape[1]
    
    for color_name, (lower, upper) in COLOR_MAPPINGS.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(img_cv, lower, upper)
        color_counts[color_name] = np.sum(mask > 0) / total_pixels
    
    return color_counts

def get_dominant_colors(image, top_n=2):
    color_dist = get_color_distribution(image)
    return [color for color, _ in color_dist.most_common(top_n)]

def analyze_color_question(question, choices, image):
    # 질문에서 특정 객체나 영역 파악
    question_lower = question.lower()
    
    # 특정 영역 분석
    if 'sky' in question_lower or '하늘' in question_lower:
        color_dist = get_color_distribution(image, region='sky')
    else:
        color_dist = get_color_distribution(image)
    
    # 가장 많이 나타나는 색상들
    dominant_colors = [color for color, freq in color_dist.most_common() if freq > 0.1]
    
    # 선택지와 매칭
    for idx, choice in enumerate(choices):
        choice_lower = choice.lower()
        for color in dominant_colors:
            if color in choice_lower:
                return idx
    
    return None

# CLIP 모델 로드
clip_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
print(f"✅ Loading model: {clip_model_name}")
clip = CLIPModel.from_pretrained(clip_model_name).to(device)
processor = CLIPProcessor.from_pretrained(clip_model_name)
clip.eval()

# 예측 함수
def predict(image_path, question, choices):
    image = Image.open(image_path).convert("RGB")
    
    # 색상 관련 질문인지 확인
    color_keywords = ['color', 'colours', '색상', '색깔']
    is_color_question = any(keyword in question.lower() for keyword in color_keywords)
    
    if is_color_question:
        # 색상 분석 수행
        color_pred = analyze_color_question(question, choices, image)
        if color_pred is not None:
            return color_pred
    
    # 색상 분석으로 답을 찾지 못했거나 다른 종류의 질문인 경우 CLIP 모델 사용
    prompts = [f"{question} 선택지: {choice}" for choice in choices]
    inputs = processor(text=prompts, images=[image]*4, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip(**inputs)
        logits = outputs.logits_per_image
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
