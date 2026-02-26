#!/bin/bash
# Stage 1: AnomalyOV encoder fine-tuning on industrial dataset
# SigLIP은 frozen, AnomalyOV만 Balanced BCE로 학습

# -----------------------------------------------------------------------
# 환경 설정
# -----------------------------------------------------------------------
export OMP_NUM_THREADS=4

# 프로젝트 루트 (이 스크립트 위치 기준 상위 2단계)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# -----------------------------------------------------------------------
# 경로 설정  (필요에 따라 수정)
# -----------------------------------------------------------------------
DATA_PATH="/path/to/your/dataset.json"       # annotation JSON 경로
IMAGE_FOLDER="/path/to/your/image/root"      # 이미지 루트 폴더

PRETRAINED_EXPERT="$PROJECT_ROOT/pretrained_expert_7b.pth"   # 7b 또는 05b
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/encoder_ft"

# SigLIP (로컬 캐시가 있다면 절대경로 사용 가능)
VISION_TOWER="google/siglip-so400m-patch14-384"

# -----------------------------------------------------------------------
# 학습 하이퍼파라미터
# -----------------------------------------------------------------------
NUM_EPOCHS=10
BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_ACCUM=1
NUM_WORKERS=4
SAVE_EVERY=2

# -----------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------
cd "$PROJECT_ROOT" || exit 1

python train_encoder.py \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --pretrained_expert "$PRETRAINED_EXPERT" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --grad_accum_steps $GRAD_ACCUM \
    --num_workers $NUM_WORKERS \
    --save_every $SAVE_EVERY \
    --bf16

echo "Done. Checkpoints in: $OUTPUT_DIR"
