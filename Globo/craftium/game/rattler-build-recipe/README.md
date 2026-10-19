# rattler-build recipe

A [rattler-build](https://prefix-dev.github.io/rattler-build/latest/) recipe.

Use the [`build.sh`](../build.sh) script to build the conda package locally.

The [`release.sh`](../release.sh) scripts can be used to release the conda package and uses the current git commit hash as the version number.

The [`release.yaml`](../.github/workflows/release.yaml) Github workflow releases the package under the x.y.z version. To trigger this workflow, change the version number in [`recipe.yaml`](./recipe.yaml) and push a commit.
