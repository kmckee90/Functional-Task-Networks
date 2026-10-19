#!/usr/bin/env bash

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

GLOBO_VERSION=$(git rev-parse --short HEAD) \
rattler-build build \
    --recipe=./rattler-build-recipe/recipe.yaml \
    --channel=https://repo.prefix.dev/obelisk-public \
    --channel=conda-forge \
    --package-format=conda \
    --experimental
