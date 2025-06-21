import re

# 정답 알파벳 추출 함수
def extract_answer_letter(text):
    match = re.search(r"Answer:\s*([A-Da-d])\b", text)
    return match.group(1).upper() if match else "?" 