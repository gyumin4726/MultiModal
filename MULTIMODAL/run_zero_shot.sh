#!/bin/bash

# Zero-shot VQA 추론 실행 스크립트
# train_vmamba_fscil.sh와 동일한 VMamba 백본을 사용하되 학습 없이 추론만 수행

echo "Starting Zero-shot VQA Inference..."
echo "Using VMamba backbone (same as train_vmamba_fscil.sh) + BERT"
echo "No training required - using pre-trained weights only"

# 기본 설정
TEST_CSV="dev/dev_test.csv"
IMAGE_DIR="dev/input_images"
OUTPUT="submission.csv"
BATCH_SIZE=8
DEVICE="cuda"
SIMILARITY_METHOD="cosine"
TEMPERATURE=1.0

# 인수가 제공된 경우 사용
if [ $# -ge 1 ]; then
    TEST_CSV=$1
fi

if [ $# -ge 2 ]; then
    IMAGE_DIR=$2
fi

if [ $# -ge 3 ]; then
    OUTPUT=$3
fi

echo "Configuration:"
echo "  Test CSV: $TEST_CSV"
echo "  Image Directory: $IMAGE_DIR"
echo "  Output File: $OUTPUT"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Similarity Method: $SIMILARITY_METHOD"
echo "  Temperature: $TEMPERATURE"
echo ""

# 추론 실행
python zero_shot_inference.py \
    --test_csv $TEST_CSV \
    --image_dir $IMAGE_DIR \
    --output $OUTPUT \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --similarity_method $SIMILARITY_METHOD \
    --temperature $TEMPERATURE

echo ""
echo "Zero-shot inference completed!"
echo "Results saved to: $OUTPUT" 