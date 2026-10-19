#!/usr/bin/env bash

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

rm -rf ./output/

./build.sh

for pkg in $(find ./output -type f \( -name "*.conda" \) ); do
    echo "Uploading ${pkg}"
    rattler-build upload prefix -c obelisk "${pkg}" || echo "Failed to upload ${pkg}"
done
