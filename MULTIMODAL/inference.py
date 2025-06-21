import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config import device
from utils import extract_answer_letter

# 추론
def run_inference(processor, model):
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
    
    return results 