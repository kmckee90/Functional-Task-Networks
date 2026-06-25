#!/usr/bin/env bash
# Clear generated artifacts: GIF peeks (artifacts/*.gif) and centroid images
# (artifacts/centroids/*.png). Keeps the centroid masses log (!masses.csv).
set -e
cd "$(dirname "$0")"
rm -f ./*.gif
rm -f ./centroids/*.png
rm -f ./centroids/*.csv
rm -f ./checkpoints/*
echo "cleared: artifacts/*.gif and artifacts/centroids/*.png"
