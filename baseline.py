import os
import re
import torch
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 환경 설정
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# 시드 고정
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# 모델 로딩
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",
    torch_dtype=torch.float16
)



# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    match = re.search(r"Answer:\s*([A-Da-d])\b", text)
    return match.group(1).upper() if match else "?"

# 추론
test = pd.read_csv('./dev_test.csv')
results = []

for _, row in tqdm(test.iterrows(), total=len(test)):
    image = Image.open(row['img_path']).convert("RGB")
    choices = [row[c] for c in ['A', 'B', 'C', 'D']]

    prompt = (
        "You are a helpful AI that answers multiple-choice questions based on the given image.\n"
        "Select the best answer from A, B, C, or D.\n\n"
        f"Question: {row['Question']}\n"
        + "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]) +
        "\nAnswer:"
    )

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: (v.half().to(device) if v.dtype == torch.float32 else v.to(device)) for k, v in inputs.items()}

    output = model.generate(**inputs, max_new_tokens=3, do_sample=False, temperature=0.0)
    decoded = processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()
    results.append(extract_answer_letter(decoded))
print('✅ Done.')

submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = results
submission.to_csv('./baseline_submit.csv', index=False)
print("✅ Done.")
