#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -jobname=SiT -output-directory=sections/SiT sections/SiT/standalone.tex
done
