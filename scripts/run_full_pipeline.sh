#!/bin/bash
# ============================================================================
# Genova -- End-to-End Training Pipeline
# ============================================================================
# Run the complete Genova training pipeline from scratch:
#   1. Download reference genome + annotations
#   2. Prepare training data (windowing, tokenization, MLM masking)
#   3. Train foundation model
#   4. Evaluate on benchmarks
#   5. Generate model card
#   6. Run data quality report
#
# Usage:
#   bash scripts/run_full_pipeline.sh [--small|--large|--mamba] [--skip-download] [--data-dir DIR]
#
# Examples:
#   bash scripts/run_full_pipeline.sh --small
#   bash scripts/run_full_pipeline.sh --large --skip-download --data-dir /data/genova
#   bash scripts/run_full_pipeline.sh --mamba --data-dir ./my_data
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_SIZE="small"
SKIP_DOWNLOAD=false
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
LOG_FILE="${PROJECT_DIR}/pipeline.log"
PYTHON="${PYTHON:-python}"
GPUS=""
WANDB_PROJECT=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --small)
            MODEL_SIZE="small"
            shift
            ;;
        --large)
            MODEL_SIZE="large"
            shift
            ;;
        --mamba)
            MODEL_SIZE="mamba"
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash scripts/run_full_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --small            Use small config (4L/256d, ~3M params) [default]"
            echo "  --large            Use large config (12L/768d, ~85M params)"
            echo "  --mamba            Use Mamba config (12L/768d SSM, ~85M params)"
            echo "  --skip-download    Skip data download if files already exist"
            echo "  --data-dir DIR     Data directory (default: ./data)"
            echo "  --output-dir DIR   Output directory (default: ./outputs)"
            echo "  --gpus N           Number of GPUs to use"
            echo "  --wandb-project P  Enable wandb logging with project name"
            echo "  --log-file FILE    Log file path (default: ./pipeline.log)"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information."
            exit 1
            ;;
    esac
done

# Resolve config path
CONFIG_FILE="${PROJECT_DIR}/configs/train_${MODEL_SIZE}.yaml"

# ---------------------------------------------------------------------------
# Colours and logging
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${msg}" | tee -a "$LOG_FILE"
}

log_step() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BOLD}${CYAN}========================================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}${CYAN}  STEP $1: $2${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}${CYAN}========================================================================${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

log_success() {
    log "${GREEN}[OK]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

step_start_time=0

start_timer() {
    step_start_time=$(date +%s)
}

elapsed_time() {
    local end_time
    end_time=$(date +%s)
    local elapsed=$(( end_time - step_start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo "${mins}m ${secs}s"
}

# ---------------------------------------------------------------------------
# Error handling and cleanup
# ---------------------------------------------------------------------------

PIPELINE_START_TIME=$(date +%s)
PIPELINE_STATUS="FAILED"
CURRENT_STEP="initialization"

cleanup() {
    local exit_code=$?
    local end_time
    end_time=$(date +%s)
    local total_elapsed=$(( end_time - PIPELINE_START_TIME ))
    local total_mins=$(( total_elapsed / 60 ))
    local total_secs=$(( total_elapsed % 60 ))

    echo "" | tee -a "$LOG_FILE"
    echo -e "${BOLD}========================================================================${NC}" | tee -a "$LOG_FILE"

    if [[ "$PIPELINE_STATUS" == "SUCCESS" ]]; then
        echo -e "${GREEN}${BOLD}  PIPELINE COMPLETED SUCCESSFULLY${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}${BOLD}  PIPELINE FAILED at step: ${CURRENT_STEP}${NC}" | tee -a "$LOG_FILE"
        echo -e "${RED}  Exit code: ${exit_code}${NC}" | tee -a "$LOG_FILE"
    fi

    echo -e "${BOLD}  Total time: ${total_mins}m ${total_secs}s${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}  Log file: ${LOG_FILE}${NC}" | tee -a "$LOG_FILE"
    echo -e "${BOLD}========================================================================${NC}" | tee -a "$LOG_FILE"
}

trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

# Initialize log file
mkdir -p "$(dirname "$LOG_FILE")"
echo "# Genova Pipeline Log - $(date)" > "$LOG_FILE"
echo "# Model: ${MODEL_SIZE}" >> "$LOG_FILE"
echo "# Data dir: ${DATA_DIR}" >> "$LOG_FILE"
echo "# Config: ${CONFIG_FILE}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

log "${BOLD}Genova Training Pipeline${NC}"
log "  Model size:    ${MODEL_SIZE}"
log "  Config:        ${CONFIG_FILE}"
log "  Data dir:      ${DATA_DIR}"
log "  Output dir:    ${OUTPUT_DIR}"
log "  Skip download: ${SKIP_DOWNLOAD}"
log "  GPUs:          ${GPUS:-auto}"
log "  Wandb:         ${WANDB_PROJECT:-disabled}"
log ""

# Verify config exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Verify Python is available
if ! command -v "$PYTHON" &>/dev/null; then
    log_error "Python not found. Set PYTHON environment variable."
    exit 1
fi

log "Python: $($PYTHON --version 2>&1)"

# Create directories
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

# =========================================================================
# STEP 1: Download reference genome + annotations
# =========================================================================

CURRENT_STEP="download"
log_step "1/6" "Download Reference Genome and Annotations"
start_timer

FASTA_FILE="${DATA_DIR}/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

if [[ "$SKIP_DOWNLOAD" == true ]] && [[ -f "$FASTA_FILE" ]]; then
    log_warn "Skipping download (--skip-download set and FASTA exists)."
else
    bash "${SCRIPT_DIR}/download_data.sh" "$DATA_DIR" 2>&1 | tee -a "$LOG_FILE"
fi

# Verify critical file exists
if [[ ! -f "$FASTA_FILE" ]] && [[ ! -L "${DATA_DIR}/hg38.fa" ]]; then
    log_error "Reference FASTA not found after download step."
    log_error "Expected: ${FASTA_FILE}"
    exit 1
fi

# Use symlink if it exists
if [[ -L "${DATA_DIR}/hg38.fa" ]]; then
    FASTA_PATH="${DATA_DIR}/hg38.fa"
elif [[ -f "$FASTA_FILE" ]]; then
    FASTA_PATH="$FASTA_FILE"
fi

log_success "Step 1 complete. ($(elapsed_time))"

# =========================================================================
# STEP 2: Prepare training data
# =========================================================================

CURRENT_STEP="prepare_data"
log_step "2/6" "Prepare Training Data"
start_timer

PREPARED_DIR="${DATA_DIR}/prepared"

# Set tokenizer params based on model size
case "$MODEL_SIZE" in
    small)
        TOKENIZER_MODE="nucleotide"
        WINDOW_SIZE=1000
        STRIDE=250
        MAX_TOKENS=1024
        KMER_SIZE=6
        ;;
    large)
        TOKENIZER_MODE="kmer"
        WINDOW_SIZE=4000
        STRIDE=1000
        MAX_TOKENS=2048
        KMER_SIZE=6
        ;;
    mamba)
        TOKENIZER_MODE="kmer"
        WINDOW_SIZE=10000
        STRIDE=2500
        MAX_TOKENS=4096
        KMER_SIZE=6
        ;;
esac

$PYTHON "${SCRIPT_DIR}/prepare_training_data.py" \
    --fasta "$FASTA_PATH" \
    --output-dir "$PREPARED_DIR" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --tokenizer-mode "$TOKENIZER_MODE" \
    --kmer-size "$KMER_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --num-workers 4 \
    2>&1 | tee -a "$LOG_FILE"

# Verify outputs
if [[ ! -f "${PREPARED_DIR}/stats.json" ]]; then
    log_error "Data preparation failed: stats.json not found."
    exit 1
fi

log_success "Step 2 complete. ($(elapsed_time))"

# =========================================================================
# STEP 3: Train foundation model
# =========================================================================

CURRENT_STEP="train"
log_step "3/6" "Train Foundation Model"
start_timer

TRAIN_CMD="$PYTHON ${SCRIPT_DIR}/train_foundation_model.py \
    --config $CONFIG_FILE \
    --data-dir $PREPARED_DIR \
    --output-dir $OUTPUT_DIR"

if [[ -n "$GPUS" ]]; then
    TRAIN_CMD="$TRAIN_CMD --gpus $GPUS"
fi

if [[ -n "$WANDB_PROJECT" ]]; then
    TRAIN_CMD="$TRAIN_CMD --wandb-project $WANDB_PROJECT"
fi

eval "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"

# Find the best model checkpoint
RUN_NAME="genova_${MODEL_SIZE}"
MODEL_DIR="${OUTPUT_DIR}/${RUN_NAME}"
BEST_MODEL="${MODEL_DIR}/best_model.pt"

if [[ ! -f "$BEST_MODEL" ]]; then
    # Try to find any checkpoint
    BEST_MODEL=$(find "$MODEL_DIR" -name "*.pt" -type f 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$BEST_MODEL" ]]; then
        log_error "No model checkpoint found in ${MODEL_DIR}."
        exit 1
    fi
    log_warn "best_model.pt not found; using: ${BEST_MODEL}"
fi

log_success "Step 3 complete. Model: ${BEST_MODEL} ($(elapsed_time))"

# =========================================================================
# STEP 4: Evaluate on benchmarks
# =========================================================================

CURRENT_STEP="evaluate"
log_step "4/6" "Evaluate on Benchmarks"
start_timer

EVAL_DIR="${OUTPUT_DIR}/${RUN_NAME}/evaluation"

$PYTHON "${SCRIPT_DIR}/evaluate_model.py" \
    --model-path "$BEST_MODEL" \
    --output-dir "$EVAL_DIR" \
    2>&1 | tee -a "$LOG_FILE"

log_success "Step 4 complete. Results: ${EVAL_DIR} ($(elapsed_time))"

# =========================================================================
# STEP 5: Generate model card
# =========================================================================

CURRENT_STEP="model_card"
log_step "5/6" "Generate Model Card"
start_timer

METRICS_FILE="${EVAL_DIR}/benchmark_report.json"
MODEL_CARD_PATH="${MODEL_DIR}/MODEL_CARD.md"

CARD_CMD="$PYTHON ${SCRIPT_DIR}/generate_model_card.py \
    --model-path $BEST_MODEL \
    --output $MODEL_CARD_PATH"

if [[ -f "$METRICS_FILE" ]]; then
    CARD_CMD="$CARD_CMD --metrics-path $METRICS_FILE"
fi

eval "$CARD_CMD" 2>&1 | tee -a "$LOG_FILE"

log_success "Step 5 complete. Model card: ${MODEL_CARD_PATH} ($(elapsed_time))"

# =========================================================================
# STEP 6: Data quality report
# =========================================================================

CURRENT_STEP="data_quality"
log_step "6/6" "Run Data Quality Report"
start_timer

QUALITY_REPORT="${OUTPUT_DIR}/${RUN_NAME}/data_quality_report.md"

$PYTHON "${SCRIPT_DIR}/run_data_quality.py" \
    --fasta "$FASTA_PATH" \
    --output "$QUALITY_REPORT" \
    --format markdown \
    2>&1 | tee -a "$LOG_FILE"

log_success "Step 6 complete. Report: ${QUALITY_REPORT} ($(elapsed_time))"

# =========================================================================
# Pipeline complete
# =========================================================================

PIPELINE_STATUS="SUCCESS"

echo "" | tee -a "$LOG_FILE"
log "${BOLD}Pipeline Artifacts:${NC}"
log "  Model checkpoint: ${BEST_MODEL}"
log "  Evaluation:       ${EVAL_DIR}/"
log "  Model card:       ${MODEL_CARD_PATH}"
log "  Data quality:     ${QUALITY_REPORT}"
log "  Training data:    ${PREPARED_DIR}/"
log "  Log file:         ${LOG_FILE}"
