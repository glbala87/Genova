#!/bin/bash
# ============================================================================
# Genova -- Download Reference Genome and Annotation Data
# ============================================================================
# Downloads real genomic data required for training and evaluating Genova.
#
# Usage:
#   bash scripts/download_data.sh [output_dir]
#
# Default output_dir: ./data
#
# Data downloaded:
#   1. GRCh38 reference genome (Ensembl, primary assembly)
#   2. ClinVar VCF (NCBI, for variant effect training/evaluation)
#   3. GENCODE v44 gene annotations (GTF)
#   4. JASPAR 2024 motif database (MEME format)
#   5. PhyloP conservation scores (UCSC, bigWig)
#
# Requirements: wget or curl, md5sum or md5, gunzip, samtools (optional)
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR="${1:-./data}"

# Subdirectories
REF_DIR="${OUTPUT_DIR}/reference"
ANN_DIR="${OUTPUT_DIR}/annotations"
VAR_DIR="${OUTPUT_DIR}/variants"
MOT_DIR="${OUTPUT_DIR}/motifs"
CON_DIR="${OUTPUT_DIR}/conservation"

# URLs -- using stable, versioned URLs for reproducibility
GRCH38_URL="https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
GRCH38_FILE="Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
GRCH38_MD5="3afa3950e92e1fd0cff03af1ce470409"

CLINVAR_URL="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
CLINVAR_TBI_URL="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi"
CLINVAR_FILE="clinvar.vcf.gz"
CLINVAR_TBI_FILE="clinvar.vcf.gz.tbi"

GENCODE_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"
GENCODE_FILE="gencode.v44.annotation.gtf.gz"
GENCODE_MD5="c89dae2fb06d5e4b2e5e7057e08bcd0b"

JASPAR_URL="https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
JASPAR_FILE="JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"

PHYLOP_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"
PHYLOP_FILE="hg38.phyloP100way.bw"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Colours for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect download tool (prefer wget for progress bars, fall back to curl)
download_file() {
    local url="$1"
    local dest="$2"
    local desc="${3:-$(basename "$dest")}"

    if [ -f "$dest" ]; then
        log_success "Already exists: ${desc}"
        return 0
    fi

    log_info "Downloading: ${desc}"
    log_info "  URL: ${url}"
    log_info "  Destination: ${dest}"

    if command -v wget &>/dev/null; then
        wget --progress=bar:force:noscroll -O "${dest}.tmp" "$url" 2>&1 || {
            log_error "wget failed for ${desc}"
            rm -f "${dest}.tmp"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "${dest}.tmp" "$url" || {
            log_error "curl failed for ${desc}"
            rm -f "${dest}.tmp"
            return 1
        }
    else
        log_error "Neither wget nor curl is available. Please install one."
        exit 1
    fi

    mv "${dest}.tmp" "$dest"
    log_success "Downloaded: ${desc}"
}

# Verify MD5 checksum
verify_md5() {
    local file="$1"
    local expected_md5="$2"
    local desc="${3:-$(basename "$file")}"

    if [ -z "$expected_md5" ]; then
        log_warn "No checksum provided for ${desc}, skipping verification."
        return 0
    fi

    log_info "Verifying checksum for ${desc}..."

    local actual_md5
    if command -v md5sum &>/dev/null; then
        actual_md5=$(md5sum "$file" | awk '{print $1}')
    elif command -v md5 &>/dev/null; then
        actual_md5=$(md5 -q "$file")
    else
        log_warn "Neither md5sum nor md5 found, skipping checksum verification."
        return 0
    fi

    if [ "$actual_md5" = "$expected_md5" ]; then
        log_success "Checksum OK: ${desc}"
        return 0
    else
        log_error "Checksum MISMATCH for ${desc}!"
        log_error "  Expected: ${expected_md5}"
        log_error "  Got:      ${actual_md5}"
        log_error "  The file may be corrupted. Delete it and retry."
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------------

log_info "============================================================"
log_info "Genova Data Download Script"
log_info "============================================================"
log_info "Output directory: ${OUTPUT_DIR}"
echo ""

for dir in "$REF_DIR" "$ANN_DIR" "$VAR_DIR" "$MOT_DIR" "$CON_DIR"; do
    mkdir -p "$dir"
done
log_success "Directory structure created."
echo ""

# ---------------------------------------------------------------------------
# 1. GRCh38 Reference Genome
# ---------------------------------------------------------------------------

log_info "--- [1/5] GRCh38 Reference Genome (Ensembl) ---"

download_file "$GRCH38_URL" "${REF_DIR}/${GRCH38_FILE}" "GRCh38 reference genome"

# Verify checksum (only for the compressed file since MD5 is for that)
if [ -f "${REF_DIR}/${GRCH38_FILE}" ]; then
    verify_md5 "${REF_DIR}/${GRCH38_FILE}" "$GRCH38_MD5" "GRCh38 FASTA (gzipped)" || true
fi

# Decompress if the uncompressed file does not exist
GRCH38_FA="${REF_DIR}/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
if [ ! -f "$GRCH38_FA" ]; then
    if [ -f "${REF_DIR}/${GRCH38_FILE}" ]; then
        log_info "Decompressing reference genome..."
        gunzip -k "${REF_DIR}/${GRCH38_FILE}"
        log_success "Reference genome decompressed."
    fi
else
    log_success "Reference genome already decompressed."
fi

# Create FASTA index if samtools is available
if [ -f "$GRCH38_FA" ] && [ ! -f "${GRCH38_FA}.fai" ]; then
    if command -v samtools &>/dev/null; then
        log_info "Creating FASTA index with samtools..."
        samtools faidx "$GRCH38_FA"
        log_success "FASTA index created."
    else
        log_warn "samtools not found. FASTA index will be created by pyfaidx at runtime."
    fi
fi

# Create a convenience symlink for Genova's default config
SYMLINK="${OUTPUT_DIR}/hg38.fa"
if [ -f "$GRCH38_FA" ] && [ ! -L "$SYMLINK" ] && [ ! -f "$SYMLINK" ]; then
    ln -s "reference/$(basename "$GRCH38_FA")" "$SYMLINK"
    log_success "Created symlink: data/hg38.fa -> reference/$(basename "$GRCH38_FA")"
fi

echo ""

# ---------------------------------------------------------------------------
# 2. ClinVar VCF
# ---------------------------------------------------------------------------

log_info "--- [2/5] ClinVar VCF (NCBI) ---"

download_file "$CLINVAR_URL" "${VAR_DIR}/${CLINVAR_FILE}" "ClinVar VCF"
download_file "$CLINVAR_TBI_URL" "${VAR_DIR}/${CLINVAR_TBI_FILE}" "ClinVar VCF index"

# ClinVar is updated frequently so we skip checksum verification
log_info "ClinVar is updated periodically; no static checksum verification."

echo ""

# ---------------------------------------------------------------------------
# 3. GENCODE Gene Annotations
# ---------------------------------------------------------------------------

log_info "--- [3/5] GENCODE v44 Gene Annotations ---"

download_file "$GENCODE_URL" "${ANN_DIR}/${GENCODE_FILE}" "GENCODE v44 GTF"

if [ -f "${ANN_DIR}/${GENCODE_FILE}" ]; then
    verify_md5 "${ANN_DIR}/${GENCODE_FILE}" "$GENCODE_MD5" "GENCODE GTF (gzipped)" || true
fi

# Decompress
GENCODE_GTF="${ANN_DIR}/gencode.v44.annotation.gtf"
if [ ! -f "$GENCODE_GTF" ]; then
    if [ -f "${ANN_DIR}/${GENCODE_FILE}" ]; then
        log_info "Decompressing GENCODE annotation..."
        gunzip -k "${ANN_DIR}/${GENCODE_FILE}"
        log_success "GENCODE annotation decompressed."
    fi
else
    log_success "GENCODE annotation already decompressed."
fi

echo ""

# ---------------------------------------------------------------------------
# 4. JASPAR Motif Database
# ---------------------------------------------------------------------------

log_info "--- [4/5] JASPAR 2024 Motif Database ---"

download_file "$JASPAR_URL" "${MOT_DIR}/${JASPAR_FILE}" "JASPAR 2024 motifs (MEME format)"

echo ""

# ---------------------------------------------------------------------------
# 5. PhyloP Conservation Scores
# ---------------------------------------------------------------------------

log_info "--- [5/5] PhyloP 100-way Conservation Scores (UCSC) ---"

download_file "$PHYLOP_URL" "${CON_DIR}/${PHYLOP_FILE}" "PhyloP 100-way bigWig"

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log_info "============================================================"
log_info "Download Summary"
log_info "============================================================"
echo ""

# Print directory sizes
for dir in "$REF_DIR" "$ANN_DIR" "$VAR_DIR" "$MOT_DIR" "$CON_DIR"; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        log_info "  $(basename "$dir"): ${size}"
    fi
done

echo ""
log_info "Total data size:"
du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print "  " $1}'

echo ""
log_success "All downloads complete!"
echo ""
log_info "Next steps:"
log_info "  1. Prepare training data:  python scripts/prepare_training_data.py --fasta ${GRCH38_FA}"
log_info "  2. Train the model:        python scripts/train_foundation_model.py --config configs/train_small.yaml"
log_info "  3. Evaluate:               python scripts/evaluate_model.py --model-path outputs/genova_pretrain/best_model.pt"
echo ""
