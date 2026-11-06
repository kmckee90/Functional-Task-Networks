#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -jobname=benchmarking -output-directory=sections/benchmarking sections/benchmarking/standalone.tex
done
