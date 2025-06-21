import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 모델 로딩
def load_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto",
        torch_dtype=torch.float16
    )
    return processor, model 