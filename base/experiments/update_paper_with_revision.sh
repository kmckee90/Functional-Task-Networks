#!/usr/bin/env bash
# update_paper_with_revision.sh — merge sharded results, regenerate
# revision tables, and rebuild the NeurIPS paper.
set -e
cd "$(dirname "$0")/.."

echo ">>> merging revision shards..."
python3 experiments/run_revision_merge.py

echo ">>> regenerating revision tables..."
python3 experiments/generate_revision_tables.py

echo ">>> regenerating revision figures..."
python3 experiments/generate_revision_figs.py

echo ">>> rebuilding NeurIPS paper..."
( cd paper/neurips && pdflatex -interaction=batchmode main.tex >/dev/null 2>&1 \
                  && pdflatex -interaction=batchmode main.tex >/dev/null 2>&1 \
                  && grep -E "Output written|^!" main.log | tail -3 )
