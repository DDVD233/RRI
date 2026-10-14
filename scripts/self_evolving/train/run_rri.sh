#!/usr/bin/env bash
# RRI training runs in the configurations reported in the paper, as named settings
# over rri_pipeline.sh. Deployment settings (teacher endpoint, keys, knowledge base,
# data and output roots) are read from the environment; see ../env.example.
#
#   bash scripts/self_evolving/train/run_rri.sh                        # RRI-9B: GPT teacher + judge, KB retrieval + web search
#   MODEL=Qwen/Qwen3.6-27B bash scripts/self_evolving/train/run_rri.sh # RRI-27B
#   SETTING=generation_only bash ...   # the Generation Only baseline: fixed simple prompts, no adversary
#   TOOLS=none JUDGE=self bash ...     # no retrieval, the frozen model grades training and validation
#   JUDGE=self_all bash ...            # every role played by the frozen model: no external API at all
#   SE_DOMAIN=my_domain.yaml VAL_PARQUET=my_val.parquet bash ...   # your own task (see the README)
#   DRY_RUN=1 bash ...                 # print the resolved commands and exit
#
# Settings:
#   SETTING   rri (default) | generation_only
#   MODEL     the policy to train (default Qwen/Qwen3.5-9B)
#   TOOLS     retrieval_web (default): the knowledge-base tool and Serper web search
#             retrieval: knowledge-base tool only
#             web:       web search only, no knowledge base needed (the non-medical runs)
#             none:      no tools
#   JUDGE     teacher (default): LLM_MODEL grades training and validation rollouts
#             self:      the frozen FROZEN_MODEL grades both
#             self_all:  the frozen model also proposes, writes rubrics and plays the adversary
#   FROZEN_BASE, FROZEN_MODEL   OpenAI-compatible endpoint of the frozen model that writes
#             evidence briefs and serves as self-judge (default Qwen/Qwen3.5-9B, started
#             on GPU $FROZEN_GPU of this node when FROZEN_BASE is not given)
#   N_GPUS    training GPUs on this node (default 4)
#   EXP       experiment name (default derived from the settings)
#
# Any knob of rri_pipeline.sh can still be set explicitly; explicit values win over the
# presets below. Extra arguments are passed to the verl trainer as Hydra overrides.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SETTING="${SETTING:-rri}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
TOOLS="${TOOLS:-retrieval_web}"
JUDGE="${JUDGE:-teacher}"

# preset KEY=value: export unless the caller already set KEY.
preset() {
    local kv
    for kv in "$@"; do
        local k="${kv%%=*}"
        [ -n "${!k+x}" ] || export "$kv"
    done
}

case "$SETTING" in
    rri)
        # Prompt evolution (meta-optimizer + exploit memo), the adversary's
        # hack-then-patch admission (probe every task, up to two repair rounds that
        # rewrite the task and rubric), and repair of exploits found in training.
        preset EVOLVE=1 SPEC_GAP=1 SPEC_GAP_SHIP=1 PROBE=1 PATCH=1 HACK_MEMO=1 \
               HB_PROBE_MODE=gate HB_REFINE_MODE=rewrite HB_PROBE_RATE=1.0 \
               HB_REFINE_ROUNDS=2 HB_REFINE_BACKGROUND=1 HB_PATCH_ROUNDS=2 \
               HB_PATCH_ASYNC=1 HB_PATCH_MAX_PER_QID=6 HB_PATCH_MIN_MARGIN=0.15 \
               HB_PATCHED_MAX_ITEMS=12 HB_PATCH_MINT_ITEMS=5 HB_REWRITE_MAX_GROW=4 \
               HB_REFINE_BG_MAX=48 HB_MEMO_MAX_CHARS=2400 ;;
    generation_only)
        # Fixed one-to-two-sentence proposer and generator prompts; no evolution,
        # no adversary. The exploit measurement stays on (it changes no gradient).
        preset EVOLVE=0 SPEC_GAP=1 SIMPLE_PROMPT=1 ;;
    *) echo "FATAL: SETTING=$SETTING (rri | generation_only)" >&2; exit 1 ;;
esac

case "$TOOLS" in
    retrieval_web) preset RETRIEVAL=1 WEB_SEARCH_TOOL=1 ;;
    retrieval)     preset RETRIEVAL=1 WEB_SEARCH_TOOL=0 ;;
    web)           preset RETRIEVAL=0 WEB_ONLY=1 WEB_SEARCH_TOOL=1 ;;
    none)          preset RETRIEVAL=0 WEB_SEARCH_TOOL=0 ;;
    *) echo "FATAL: TOOLS=$TOOLS (retrieval_web | retrieval | web | none)" >&2; exit 1 ;;
esac
# The paper's runs ground the solver through its own tools only; the GPT-mediated
# web lookup inside /retrieve stays off.
preset WEB_EVIDENCE=0

case "$JUDGE" in
    teacher)  ;;
    self)     preset SELF_JUDGE=1 VAL_SELF_JUDGE=1 ALLOW_EVOLVE_SELF_JUDGE=1 ;;
    self_all) preset SELF_JUDGE=1 VAL_SELF_JUDGE=1 ALLOW_EVOLVE_SELF_JUDGE=1 SELF_ALL=1 ;;
    *) echo "FATAL: JUDGE=$JUDGE (teacher | self | self_all)" >&2; exit 1 ;;
esac

# The frozen model: evidence summarizer for the knowledge-base tool and self-judge.
preset SUMM_BASE="${FROZEN_BASE:-http://localhost:8199/v1}" \
       SUMM_MODEL="${FROZEN_MODEL:-Qwen/Qwen3.5-9B}" SUMM_FALLBACK_BASE=""

# Built-in non-medical domains: no medical knowledge-base anchors or style seeds,
# no length rule in their official score, and the medical referee smoke test skipped.
case "${SE_DOMAIN:-medical}" in
    prbench|profbench)
        preset HB_KB_ANCHOR_SHARE=0 HB_STYLE_SEED_SHARE=0 HB_VAL_LENGTH_PENALTY_PER_500=0 \
               HB_LENGTH_CENTER=4000 REFEREE_SMOKE_SOFT=1 ;;
esac

preset ACTOR_MODEL_PATH="$MODEL" STEPS=100000
_size="$(basename "$MODEL" | tr '[:upper:].' '[:lower:]_')"
preset EXP="rri_${_size}_${SETTING}_${TOOLS}_${JUDGE}${SE_DOMAIN:+_$(basename "${SE_DOMAIN%.*}")}"

exec bash "$HERE/rri_pipeline.sh" "$@"
