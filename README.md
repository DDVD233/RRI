# RRI: Recursive Reward Improvement via Adversarial Meta-Prompt Evolution

RRI trains a model on open-ended expert tasks, such as clinical consults, documentation and evidence questions, with **no training data**. A proposer writes each training task together with the rubric that grades it, and a solver learns from those tasks with reinforcement learning (GRPO).

Model-written rubrics are easy to game: a solver can name the expected topics without the substance behind them. RRI therefore supervises the reward itself with an adversary:

1. **Hack.** Before a task is served, an adversary writes an answer that games the rubric while being as useless as possible. The same model also writes an honest answer without seeing the rubric.
2. **Patch.** If the gaming answer wins, up to two repair steps rewrite the task and rubric so that the honest answer comes out ahead.
3. **Evolve.** The exploits found during admission and during training are summarized into a memo. A meta-optimizer then rewrites the prompts that generate future tasks and rubrics.

On HealthBench Professional, RRI improves the length-adjusted score over the *Generation Only* self-improvement baseline:
- 10.9 points at 9B;
- 12.4 points at 27B;
- 10.3 points with no external model at all.

The 27B model trained only on the HealthBench Professional *description* also improves by 6.6 points on PRBench Hard and by 8.0 points on ProfBench, without seeing a task from either.

**Paper:** arXiv link coming soon.

> Research code and models. The models are not medical devices and must not be used for clinical decisions.

## Contents

- [Results](#results)
- [Models](#models)
- [Getting started](#getting-started)
  - [0. Install](#0-install)
  - [1. Configure endpoints and keys](#1-configure-endpoints-and-keys)
  - [2. Prepare data](#2-prepare-data)
  - [3. Build the clinical knowledge base](#3-build-the-clinical-knowledge-base)
  - [4. Run the evolving pipeline](#4-run-the-evolving-pipeline)
  - [5. Evaluate](#5-evaluate)
- [Repository layout](#repository-layout)
- [Acknowledgements and license](#acknowledgements-and-license)

## Results

HealthBench Professional (525 physician-written tasks), official length-adjusted rubric score. *Generation Only* is self-improvement with an untuned proposer prompt and no adversary. All rows train on generated tasks only.

| Setting (policy, teacher and judge, tools) | Untrained | Generation Only | **RRI** |
|---|---:|---:|---:|
| Qwen3.5-9B, GPT teacher and judge, knowledge base + web search | 0.329 | 0.424 | **0.533** |
| Qwen3.6-27B, GPT teacher and judge, knowledge base + web search | 0.374 | 0.392 | **0.516** |
| Qwen3.5-9B, every role played by the frozen 9B, knowledge base + web search | 0.348 | 0.418 | **0.521** |
| Qwen3.5-9B, frozen-9B judge, no tools | 0.244 | 0.381 | **0.421** |

## Models

| Model | Base | Trained for | Weights | Score of these weights |
|---|---|---|---|---|
| **Qwen3.6-RRI-27B** | Qwen3.6-27B | HealthBench Professional description; GPT teacher and judge; knowledge base + web search tools | [ddvd233/Qwen3.6-RRI-27B](https://huggingface.co/ddvd233/Qwen3.6-RRI-27B) | HealthBench Professional 0.485 (untrained 0.374) |
| Qwen3.5-RRI-9B (no tools) | Qwen3.5-9B | Same description; the frozen 9B grades; no tools | [ddvd233/Qwen3.5-RRI-9B](https://huggingface.co/ddvd233/Qwen3.5-RRI-9B) | HealthBench Professional 0.391 (untrained 0.244) |
| **Qwen3.6-RSIMed-27B** | Qwen3.6-27B | Rare-disease diagnosis from MIMIC-IV admissions (companion paper RSIMed) | [ddvd233/Qwen3.6-RSIMed-27B](https://huggingface.co/ddvd233/Qwen3.6-RSIMed-27B) | MIMIC-IV rare-disease diagnostic accuracy 0.516 |
| Qwen3.5-RSIMed-9B | Qwen3.5-9B | Same task | [ddvd233/Qwen3.5-RSIMed-9B](https://huggingface.co/ddvd233/Qwen3.5-RSIMed-9B) | MIMIC-IV rare-disease diagnostic accuracy 0.469 |

All four are merged bf16 Hugging Face checkpoints, served directly by vLLM or transformers. The released weights are the last surviving checkpoint of each run. Checkpoints that were not kept can score differently from these, which is why the RRI rows differ from the table above.

The RRI models were trained to answer with the tools they had during training, so evaluate them with the same tools ([step 5](#5-evaluate)).

**RSIMed** (Recursive Self-Improvement in Medicine) is our companion work on self-evolving training data for diagnosis from real clinical records. Its data-generation pipeline is not part of this repository.

## Getting started

RRI needs no training set. You provide:

- **a description of the target use**: what kind of requests, from whom, and what a good answer gets right. It is a short text and never contains evaluation items ([2.3](#23-describe-your-target-domain)).
- **a held-out evaluation set** with rubrics, used only for validation ([2.1](#21-evaluation-sets)–[2.2](#22-your-own-evaluation-set)).
- **a teacher model** behind an OpenAI-compatible API ([step 1](#1-configure-endpoints-and-keys)). It writes tasks and rubrics, plays the adversary, and judges. With `JUDGE=self_all` a frozen open model plays every role instead.
- **optionally, a knowledge base** for grounding tasks and for the solver's retrieval tool ([step 3](#3-build-the-clinical-knowledge-base)), and a [Serper](https://serper.dev) key for the web-search tool.

**Hardware used in the paper:**
- **9B policy:** 4 NVIDIA B200 GPUs. With 80 GB GPUs, lower `VLLM_GPU_UTIL` or the response length.
- **27B policy:** 4 B200 GPUs and at least 800 GB of host RAM, because parameters and optimizer state are offloaded to the CPU.
- **Frozen Qwen3.5-9B** (evidence summarizer and self-judge): one more GPU, or a slice of a training GPU.
- **Milvus:** a CPU service. The embedding model needs about a third of one GPU.

### 0. Install

The tested software stack is vLLM 0.22, PyTorch 2.11, CUDA 13 and Python 3.12 (`docker/Dockerfile.rri`).

```bash
git clone https://github.com/DDVD233/RRI.git && cd RRI

# Docker (recommended)
docker build -f docker/Dockerfile.rri -t rri .

# or into an environment that already has vLLM >= 0.22
pip install -r requirements.txt -r requirements-rri.txt
pip install -e . --no-deps
```

All commands below run from the repository root.

### 1. Configure endpoints and keys

```bash
cp scripts/self_evolving/env.example scripts/self_evolving/.env   # gitignored
```

Fill in at least `LLM_MODEL` and `LLM_API_KEY` (or export `OPENAI_API_KEY`). The launchers read this file; variables already set in your shell take precedence. The file documents every setting.

| Setting | Meaning |
|---|---|
| `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` | Teacher and judge endpoint, any OpenAI-compatible chat API. The paper used GPT-5.5-Chat. |
| `LLM_PROVIDER` | `openai` sends the GPT-5-family request shape; `vllm` targets a local vLLM server. |
| `FROZEN_BASE`, `FROZEN_MODEL` | Frozen model for evidence briefs and self-judging. When `FROZEN_BASE` is unset, it is started locally. |
| `KB`, `EMBED_BASE`, `MILVUS_URI`, `MILVUS_TOKEN`, `MILVUS_COLLECTION` | Knowledge base. Set `KB=0` to run without one. |
| `SERPER_API_KEY` | Web-search tool. |
| `RRI_DATA`, `RRI_OUTPUT` | Where evaluation sets are read from and where logs, checkpoints and caches go (default `data/`, `outputs/`). |
| `WANDB_API_KEY` | Optional. Without it, logs go to the console only. |

Keep keys in `.env` or in the environment, never in committed files.

### 2. Prepare data

#### 2.1 Evaluation sets

Every evaluation set becomes a parquet file with one rubric-graded task per row; the commands below write it.

```bash
# HealthBench Professional (the paper's benchmark): data/healthbench_pro_val.parquet
python scripts/self_evolving/eval/preprocess_healthbench_professional.py --out data/healthbench_pro_val.parquet
```

| Benchmark | Source | Command |
|---|---|---|
| HealthBench Professional | [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional) | `eval/preprocess_healthbench_professional.py --out data/healthbench_pro_val.parquet` |
| HealthBench Hard / Consensus | [openai/simple-evals](https://github.com/openai/simple-evals) | Download the JSONL to `data/bench_raw/healthbench/`, then `eval/preprocess_healthbench_hard.py --subset hard --out data/healthbench_hard_val.parquet` |
| PRBench (finance, legal) | [ScaleAI/PRBench](https://huggingface.co/datasets/ScaleAI/PRBench) | `huggingface-cli download ScaleAI/PRBench --repo-type dataset --local-dir data/bench_raw/PRBench`, then `eval/preprocess_prbench.py --subset hard` |
| ProfBench | [nvidia/ProfBench](https://huggingface.co/datasets/nvidia/ProfBench) | `huggingface-cli download nvidia/ProfBench --repo-type dataset --local-dir data/bench_raw/ProfBench`, then `eval/preprocess_profbench.py` |
| MedXpertQA (multiple choice, scored as a one-criterion rubric) | [TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | `huggingface-cli download TsinghuaC3I/MedXpertQA --repo-type dataset --local-dir data/bench_raw/MedXpertQA`, then `eval/preprocess_medxpertqa.py --subset text --out data/medxpertqa_text_val.parquet` |

All scripts are under `scripts/self_evolving/`. The HealthBench Hard script's docstring lists the source URLs.

#### 2.2 Your own evaluation set

Write one task per line as JSON:

```json
{"id": "case-001",
 "prompt": "A 64-year-old on apixaban for atrial fibrillation needs a colonoscopy with possible polypectomy next week. eGFR is 55. How should anticoagulation be handled?",
 "rubric": [
   {"criterion": "States that apixaban should be held for about 48 hours before a high-bleeding-risk procedure at this kidney function", "points": 8},
   {"criterion": "States that bridging with heparin is not needed", "points": 6},
   {"criterion": "Recommends bridging with low-molecular-weight heparin", "points": -6}],
 "category": "treatment"}
```

- `prompt` may be replaced by `conversation`: a list of `{"role", "content"}` turns ending on a user turn.
- Positive points reward content the answer should contain.
- Negative points describe an error, and are deducted when the answer makes it.
- The task score is the sum of the points of met criteria divided by the sum of the positive points.

Convert the file and point the pipeline at it:

```bash
python scripts/self_evolving/eval/prepare_rubric_dataset.py \
    --input my_tasks.jsonl --name my_bench --out data/my_bench_val.parquet
export VAL_PARQUET=data/my_bench_val.parquet        # comma-separate several sets
```

Validation reports `val-core/<name>/<category>/...` per category, and `val-core/overall/acc/mean` overall.

HealthBench's score includes a length term: 0.0147 per 500 characters, relative to 2,000 characters. Benchmarks without such a rule should set `HB_VAL_LENGTH_PENALTY_PER_500=0`.

#### 2.3 Describe your target domain

The proposer is told what the model is *for*, never what the evaluation items are. Built-in descriptions are selected with `SE_DOMAIN`:
- `medical`: HealthBench Professional, the paper's main runs.
- `general`: general medical use, naming no benchmark.
- `prbench`, `profbench`, `medxpert`.

For your own target, write a YAML file that starts from a built-in description and overrides it. [`scripts/self_evolving/domain_examples/nephrology_consults.yaml`](scripts/self_evolving/domain_examples/nephrology_consults.yaml) is a complete example:

```yaml
base: general                    # start from a built-in bundle
NAME: nephrology_consults
BENCH_NAME: nephrology consult support
DATASET_BRIEF: >-
  TARGET USE (general description; there are NO example items): an assistant that hospital
  clinicians consult about kidney problems ... A good answer commits to a specific assessment,
  names the thresholds, doses and monitoring intervals that decide the case ...
USE_CASES: {diagnosis: 0.30, treatment: 0.35, drug_dosing: 0.20, documentation: 0.15}
USE_CASE_DESC: {...}             # one sentence per use case
SPECIALTIES: [nephrology, critical_care, internal_medicine, transplant_medicine, pharmacology]
```

```bash
export SE_DOMAIN=scripts/self_evolving/domain_examples/nephrology_consults.yaml
```

The keys are documented in `scripts/self_evolving/domains.py`. A non-medical domain should also set `TOOLS=web` (or `none`) and `HB_KB_ANCHOR_SHARE=0`, because the knowledge base is medical.

#### 2.4 Optional: proposer style seeds

For the medical domain, the proposer imitates the register of real consumer and clinician questions. It never copies their content; a leak check rejects tasks that do. The seeds come from HealthSearchQA, MedQuAD, augmented clinical notes and K-QA:

```bash
python scripts/self_evolving/kb/fetch_seed_corpora.py --out data/seed_corpora/style_exemplars.jsonl
export SEED_EXEMPLARS_PATH=data/seed_corpora/style_exemplars.jsonl
```

Without seeds, tasks are generated unseeded.

### 3. Build the clinical knowledge base

The knowledge base grounds generated tasks in medical sources and serves the solver's retrieval tool. It is a Milvus collection (`medical_knowledge_v2`) with one row per passage:
- **Embedding:** [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), 2048 dimensions, cosine similarity.
- **Ranking:** search over-fetches 8×, then re-ranks by cosine × a per-source prior.
- **Diversity:** at most 3 passages per source.

The paper's knowledge base:

| Source | Passages | Download |
|---|---:|---|
| MedRAG Wikipedia | 29.9M | [MedRAG/wikipedia](https://huggingface.co/datasets/MedRAG/wikipedia) |
| MedRAG PubMed abstracts | 23.9M | [MedRAG/pubmed](https://huggingface.co/datasets/MedRAG/pubmed) |
| StatPearls clinical reviews | 162k | [NCBI Bookshelf](https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz) |
| MedRAG textbooks | 126k | [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) |
| DailyMed FDA drug labels (dosing, indications, contraindications, warnings) | 81k | [DailyMed](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm) |
| ICD-10-CM 2026 code tables | 6.7k | [CMS](https://www.cms.gov/medicare/coding-billing/icd-10-codes) |
| MedlinePlus health topics | 1.9k | [MedlinePlus XML](https://medlineplus.gov/xml.html) |

Each source has its own terms of use; check them before redistributing derived data.

A useful knowledge base can be much smaller: the four curated sources plus MedRAG textbooks are about 380k passages. Start there, and add PubMed and Wikipedia (about 54M passages to embed) if you need them.

**3.1 Start Milvus and the embedding server.**

```bash
cd scripts/self_evolving/kb/milvus && MILVUS_DATA=/path/with/space docker compose up -d && cd -
CUDA_VISIBLE_DEVICES=0 MEM=0.3 bash scripts/self_evolving/serve/start_embedding_server.sh   # :18001
```

Milvus starts with authentication on and the default `root:Milvus` login. Change the password before exposing the port (`kb/milvus/docker-compose.yml` shows how), and set `MILVUS_TOKEN` accordingly.

**3.2 Download and ingest.**

```bash
export MILVUS_URI=http://localhost:19530 MILVUS_TOKEN=root:Milvus EMBED_API_BASE=http://localhost:18001/v1
bash scripts/self_evolving/kb/download_sources.sh            # curated sources -> data/kb_sources
bash scripts/self_evolving/kb/download_sources.sh medrag     # + MedRAG; MEDRAG_CORPORA=textbooks for the small one

for src in statpearls dailymed medlineplus icd10cm medrag_textbook; do
    python scripts/self_evolving/kb/ingest_sources.py --source $src --create_collection
done
# optional, large: --source medrag_pubmed, --source medrag_wiki
```

Ingestion details:
- **Chunking:** the curated sources are cut into sentence-aware chunks of about 1,500 characters, each prefixed with its article and section heading. MedRAG snippets are ingested as published.
- **Resuming:** ingestion resumes where it stopped. `--dry_run` prints chunk statistics without writing.

**3.3 Add your own documents.** Put guidelines, protocols or notes into JSON lines with `text` and optional `title` and `id`:

```bash
python scripts/self_evolving/kb/ingest_sources.py --source jsonl --name my_guidelines \
    --input 'my_corpus/*.jsonl' --create_collection
export RETRIEVE_CURATED_SOURCES=my_guidelines               # return full-length passages
export RETRIEVE_SOURCE_WEIGHTS='{"my_guidelines": 1.0}'     # ranking prior (unlisted sources: 0.75)
```

**3.4 Optional: citations for PubMed passages.**

```bash
python scripts/self_evolving/kb/build_pubmed_titles.py
python scripts/self_evolving/kb/build_pubmed_meta.py
```

These build `data/kb/pubmed_titles.sqlite`, so retrieved abstracts carry their title, journal and year.

**3.5 The tool path at training time.** The solver's retrieval tool calls the generation server's `/retrieve` endpoint. That endpoint runs up to four queries, merges and de-duplicates the passages, and has the frozen Qwen3.5-9B write an evidence brief, which keeps facts verbatim and lists sources. The web-search tool returns raw [Serper](https://serper.dev) results through a local cache. The launcher starts both services and checks them before training.

### 4. Run the evolving pipeline

```bash
# RRI-9B as in the paper: GPT teacher and judge, knowledge-base + web-search tools
bash scripts/self_evolving/train/run_rri.sh

MODEL=Qwen/Qwen3.6-27B bash scripts/self_evolving/train/run_rri.sh       # RRI-27B
SETTING=generation_only bash scripts/self_evolving/train/run_rri.sh      # the baseline
JUDGE=self_all bash scripts/self_evolving/train/run_rri.sh               # no external model
TOOLS=none JUDGE=self KB=0 bash scripts/self_evolving/train/run_rri.sh   # smallest setup: no KB, no web
DRY_RUN=1 bash scripts/self_evolving/train/run_rri.sh                    # print the resolved commands
```

`run_rri.sh` names the paper's configurations:

| Setting | Values |
|---|---|
| `SETTING` | `rri` (default): adversarial admission, training-time exploit repair, prompt evolution. `generation_only`: fixed simple prompts, no adversary. |
| `MODEL` | Policy to train (default `Qwen/Qwen3.5-9B`). |
| `TOOLS` | `retrieval_web` (default): knowledge base + web search. `retrieval`, `web`, or `none`. |
| `JUDGE` | `teacher` (default): `LLM_MODEL` grades training and validation. `self`: the frozen model grades both. `self_all`: the frozen model plays every role. |
| `SE_DOMAIN`, `VAL_PARQUET` | Your domain description and evaluation set ([step 2](#2-prepare-data)). |
| `N_GPUS`, `FROZEN_BASE`, `FROZEN_GPU` | Placement. Without `FROZEN_BASE`, the frozen 9B shares GPU `FROZEN_GPU` (default 3) with training. |

Every knob of the underlying [`rri_pipeline.sh`](scripts/self_evolving/train/rri_pipeline.sh) can be overridden from the environment: `STEPS`, `TEST_FREQ` (5), `SAVE_FREQ` (20), `TRAIN_BS` (32), `LR` (1e-6), `MAX_RESP_LEN` (8192), `ROLLOUT_TP` (2), and the adversary knobs `HB_*`. Its header comments explain each switch. Extra arguments are passed to verl as Hydra overrides.

A run does the following, in order:
- starts the generation server (proposer, rubric writer, adversary, repair, meta-optimizer, `/retrieve`) and the frozen model and caches its tools need;
- checks every endpoint;
- fills a pool of admitted tasks;
- trains with GRPO (8 rollouts per task, no KL), validating every `TEST_FREQ` steps on the held-out set.

| Output | Location |
|---|---|
| Logs | `outputs/logs/` (`gen_server_<exp>.log`, rollouts, validation generations) |
| Evolved prompts and exploit memo | `outputs/logs/<exp>/prompts/` |
| Checkpoints | `outputs/checkpoints/<exp>/` |

Restarting with the same experiment name resumes from the last checkpoint.

To merge a checkpoint for serving:

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir outputs/checkpoints/<exp>/global_step_<N>/actor --target_dir merged/<exp>_<N>
```

### 5. Evaluate

`eval_rri.sh` evaluates any model (a Hugging Face id or a local directory) under the paper's validation protocol:
- the same tools as in training;
- temperature 1, with thinking on;
- the per-criterion judge, three votes per criterion.

```bash
MODEL=ddvd233/Qwen3.6-RRI-27B \
    bash scripts/self_evolving/eval/eval_rri.sh                                     # with tools
MODEL=Qwen/Qwen3.5-9B TOOLS=none bash scripts/self_evolving/eval/eval_rri.sh       # no tools
```

To rescore the saved answers under HealthBench's official grader (GPT-5.4, low reasoning effort):

```bash
python scripts/self_evolving/analysis/regrade_hbpro_dumps.py \
    --dump outputs/logs/val_generations/<exp>/<step>.jsonl \
    --val-parquet data/healthbench_pro_val.parquet \
    --model gpt-5.4-2026-03-05 --effort low --out outputs/regrade/<exp>_<step>.json
```

For a tool-free run of the official simple-evals harness against any vLLM-served model, use `scripts/self_evolving/eval/healthbench_professional_eval.py`.

## Repository layout

```
scripts/self_evolving/
  generation_server.py     proposer, rubric writer, adversary (hack / patch), exploit memo,
                           meta-optimizer, /retrieve; serves tasks to the trainer
  domains.py               target-use descriptions (SE_DOMAIN); domain_examples/ for your own
  self_evolving_dataset.py trainer-side client that streams generated tasks
  train/                   run_rri.sh (paper settings), rri_pipeline.sh (everything), tool configs
  eval/                    eval_rri.sh, benchmark preprocessing, prepare_rubric_dataset.py
  kb/                      knowledge base: ingestion, ranking, web evidence, Milvus compose file
  serve/                   embedding server, frozen summarizer / self-judge
  analysis/                official-grader regrading, referee check
verl/                      verl with the RRI hooks:
  utils/reward_score/      healthbench_pro.py (rubric grading), spec_gap.py (exploit detection and
                           repair hooks), reward_evolution.py, retrieval_coverage.py, self_evolving.py
  trainer/ppo/ray_trainer.py   evolution, admission and repair hooks in the training loop
  experimental/agent_loop/     retrieval_tool_agent_loop.py: multi-turn tool use with thinking budgets
  tools/                   knowledge-base retrieval and web-search tools
tests/                     CPU tests: pytest tests scripts/self_evolving/tests
```

## Acknowledgements and license

This repository is built on [verl](https://github.com/verl-project/verl). `LICENSE` (Apache-2.0) and `Notice.txt` are carried over from verl unchanged. We thank the authors of HealthBench, PRBench, ProfBench, MedXpertQA, MedRAG, and the Qwen models. Datasets and model weights keep their own licenses.

## Citation

The BibTeX entry will be added with the arXiv release.
