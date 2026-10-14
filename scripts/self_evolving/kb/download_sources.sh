#!/usr/bin/env bash
# Download (and unpack) the knowledge-base sources that kb/ingest_sources.py reads.
#
#   bash scripts/self_evolving/kb/download_sources.sh            # the four curated sources
#   bash scripts/self_evolving/kb/download_sources.sh medrag     # + MedRAG textbooks/pubmed/wikipedia
#
# Curated clinical references (public-domain / open-access US government or OA sources):
#
#   statpearls   ~10k peer-reviewed clinical review articles (NCBI Bookshelf OA)
#   dailymed     FDA drug labels -> dosing / contraindications / warnings (~19 GB)
#   medlineplus  NIH consumer-health topic summaries (patient-facing register)
#   icd10cm      the full 2026 ICD-10-CM code -> description table
#
# MedRAG corpora (Xiong et al., 2024; https://huggingface.co/MedRAG), pre-chunked:
#
#   textbooks    18 medical textbooks, ~126k snippets (small; start here)
#   pubmed       PubMed abstracts, ~24M snippets (~55 GB)
#   wikipedia    Wikipedia, ~30M snippets (large; general domain, weighted down at retrieval)
#
# The MedlinePlus XML file name carries its publication date; if the URL below 404s,
# take the current "Health Topics XML" link from https://medlineplus.gov/xml.html.
# Re-running skips files that already exist.
set -uo pipefail

OUT_DIR="${OUT_DIR:-${KB_SOURCES_DIR:-data/kb_sources}}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR" || exit 1

# fetch <url> <outfile> — skip if a non-empty file is already there.
fetch() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "SKIP $out ($(du -h "$out" | cut -f1) already present)"
    return 0
  fi
  echo "GET  $out <- $url"
  if curl -sfL --retry 3 --retry-delay 5 -o "$out.part" "$url"; then
    mv "$out.part" "$out"
    echo "DONE $out ($(du -h "$out" | cut -f1))"
  else
    echo "FAIL $out"
    rm -f "$out.part"
  fi
}

fetch "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip" icd10cm_2026.zip
fetch "https://medlineplus.gov/xml/mplus_topics_2026-07-25.xml" medlineplus_topics.xml
fetch "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz" statpearls.tar.gz

# DailyMed human prescription labels ship as 6 parts, ~3.2 GB each (read as zips).
for i in 1 2 3 4 5 6; do
  fetch "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part${i}.zip" "dm_rx_part${i}.zip"
done

# Unpack into the layout ingest_sources.py expects.
if [[ -s icd10cm_2026.zip && ! -d icd10 ]]; then
  unzip -q -o icd10cm_2026.zip -d icd10 && echo "UNPACKED icd10/"
  # The ingester reads icd10/icd10cm_order_2026.txt; flatten a nested archive layout.
  [[ -f icd10/icd10cm_order_2026.txt ]] || \
    find icd10 -name 'icd10cm_order_2026.txt' -exec cp {} icd10/ \;
fi
if [[ -s statpearls.tar.gz && ! -d statpearls ]]; then
  mkdir -p statpearls && tar -xzf statpearls.tar.gz -C statpearls && echo "UNPACKED statpearls/"
fi

if [[ "${1:-}" == "medrag" ]]; then
  command -v huggingface-cli >/dev/null || { echo "FATAL: pip install huggingface_hub" >&2; exit 1; }
  for corpus in ${MEDRAG_CORPORA:-textbooks pubmed wikipedia}; do
    echo "GET  medrag/$corpus <- huggingface.co/datasets/MedRAG/$corpus"
    huggingface-cli download "MedRAG/$corpus" --repo-type dataset \
      --include 'chunk/*' --local-dir "medrag/$corpus" >/dev/null && echo "DONE medrag/$corpus"
  done
fi

echo "ALL_DOWNLOADS_DONE in $PWD"
