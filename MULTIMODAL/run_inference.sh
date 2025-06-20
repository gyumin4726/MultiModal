#!/bin/bash

# SCPC AI Challenge 2025 - MultiModal VQA Inference Script
# 학습된 모델로 테스트 데이터 예측

echo "=== SCPC AI Challenge 2025 - MultiModal VQA Inference ==="

# 기본 설정
CONFIG=${1:-"competition"}
MODEL_PATH=${2:-"checkpoints/best_model.pth"}
TEST_CSV=${3:-"data/dev_test.csv"}
TEST_IMAGE_DIR=${4:-"data/input_images"}
OUTPUT_CSV=${5:-"baseline_submit.csv"}

echo "Configuration: $CONFIG"
echo "Model Path: $MODEL_PATH"
echo "Test CSV: $TEST_CSV"
echo "Test Image Dir: $TEST_IMAGE_DIR"
echo "Output CSV: $OUTPUT_CSV"

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 필수 파일 확인
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found: $MODEL_PATH"
    echo "Please train the model first or provide correct model path."
    exit 1
fi

if [ ! -f "$TEST_CSV" ]; then
    echo "Error: Test CSV not found: $TEST_CSV"
    exit 1
fi

if [ ! -d "$TEST_IMAGE_DIR" ]; then
    echo "Error: Test image directory not found: $TEST_IMAGE_DIR"
    exit 1
fi

# sample_submission.csv 확인
if [ ! -f "./sample_submission.csv" ]; then
    echo "Warning: sample_submission.csv not found in current directory"
    echo "The script will create submission from scratch"
fi

# 추론 실행
echo "Starting inference..."
python inference.py \
    --config $CONFIG \
    --model_path $MODEL_PATH \
    --test_csv $TEST_CSV \
    --test_image_dir $TEST_IMAGE_DIR \
    --output_csv $OUTPUT_CSV \
    --sample_submission ./sample_submission.csv \
    --batch_size 32 \
    --encoding_mode unified \
    --num_workers 4

echo "Inference completed!"

# 결과 확인
if [ -f "$OUTPUT_CSV" ]; then
    echo "✅ Submission file created: $OUTPUT_CSV"
    echo "File size: $(wc -l < $OUTPUT_CSV) lines"
    
    echo "First 5 predictions:"
    head -6 $OUTPUT_CSV
    
    echo "Answer distribution:"
    tail -n +2 $OUTPUT_CSV | cut -d',' -f2 | sort | uniq -c | sort -nr
    
    # 파일 형식 검증
    echo "Validating submission format..."
    python -c "
import pandas as pd
try:
    df = pd.read_csv('$OUTPUT_CSV')
    required_cols = ['ID', 'answer']
    if all(col in df.columns for col in required_cols):
        valid_answers = df['answer'].isin(['A', 'B', 'C', 'D']).all()
        if valid_answers:
            print('✅ Submission format is valid!')
        else:
            print('❌ Invalid answers found! Only A, B, C, D are allowed.')
    else:
        print('❌ Missing required columns: ID, answer')
except Exception as e:
    print(f'❌ Error reading submission file: {e}')
"
else
    echo "❌ Submission file not created!"
    exit 1
fi

echo "Inference script completed successfully!" 