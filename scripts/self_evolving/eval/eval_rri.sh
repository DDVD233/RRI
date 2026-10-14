#!/usr/bin/env bash
# Evaluate one model (a Hugging Face id or a local HF directory) on a held-out rubric
# set under the paper's in-loop validation protocol: rri_pipeline.sh in trainer.val_only
# mode, so the model answers every task once with the same tools, decoding (temperature
# 1, thinking on) and per-criterion judge as validation during training. Nothing is
# trained and no checkpoint is written.
#
#   MODEL=ddvd233/Qwen3.6-RRI-27B \
#       bash scripts/self_evolving/eval/eval_rri.sh                    # with KB + web search tools
#   MODEL=Qwen/Qwen3.5-9B TOOLS=none bash scripts/self_evolving/eval/eval_rri.sh
#
# Settings: MODEL (required), TOOLS (retrieval_web | retrieval | web | none, default
# retrieval_web), VAL_PARQUET (default $RRI_DATA/healthbench_pro_val.parquet), N_GPUS,
# EXP. The judge is LLM_MODEL (three votes per criterion); deployment settings as for
# training (../env.example). With the KB tool on and no FROZEN_BASE, the frozen
# summarizer is started on GPU $FROZEN_GPU (default 3), so use N_GPUS=3 on a 4-GPU node.
#
# The per-task generations land in $LOGDIR/val_generations/$EXP/; regrade them under
# another judge with scripts/self_evolving/analysis/regrade_hbpro_dumps.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:?set MODEL=<hugging face id or local HF directory>}"
TOOLS="${TOOLS:-retrieval_web}"
case "$TOOLS" in
    retrieval_web) RETRIEVAL=1; WEB_SEARCH_TOOL=1; WEB_ONLY=0 ;;
    retrieval)     RETRIEVAL=1; WEB_SEARCH_TOOL=0; WEB_ONLY=0 ;;
    web)           RETRIEVAL=0; WEB_SEARCH_TOOL=1; WEB_ONLY=1 ;;
    none)          RETRIEVAL=0; WEB_SEARCH_TOOL=0; WEB_ONLY=0 ;;
    *) echo "FATAL: TOOLS=$TOOLS (retrieval_web | retrieval | web | none)" >&2; exit 1 ;;
esac
if [ "$RETRIEVAL" = 1 ]; then _gpus=3; else _gpus=4; fi
_name="$(basename "$MODEL" | tr '[:upper:].' '[:lower:]_')"

exec env RETRIEVAL="$RETRIEVAL" WEB_SEARCH_TOOL="$WEB_SEARCH_TOOL" WEB_ONLY="$WEB_ONLY" \
     WEB_EVIDENCE=0 EVOLVE=0 SPEC_GAP=0 SPEC_GAP_SHIP=0 PROBE=0 PATCH=0 HACK_MEMO=0 \
     VAL_ONLY=1 STEPS=1 TRAIN_BS="${TRAIN_BS:-16}" N_GPUS="${N_GPUS:-$_gpus}" \
     ACTOR_MODEL_PATH="$MODEL" HB_JUDGE_VOTES="${HB_JUDGE_VOTES:-3}" \
     SUMM_BASE="${FROZEN_BASE:-${SUMM_BASE:-http://localhost:8199/v1}}" \
     SUMM_MODEL="${FROZEN_MODEL:-${SUMM_MODEL:-Qwen/Qwen3.5-9B}}" SUMM_FALLBACK_BASE="" \
     FROZEN_GPU="${FROZEN_GPU:-3}" GEN_PORT="${GEN_PORT:-8061}" \
     EXP="${EXP:-eval_${_name}_${TOOLS}}" \
     bash "$HERE/../train/rri_pipeline.sh" "$@"
