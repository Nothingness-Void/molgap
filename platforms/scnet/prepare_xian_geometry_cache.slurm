#!/usr/bin/env bash
#SBATCH --job-name=molgap-xian-cache-accept
#SBATCH --partition=xahdtest
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

: "${ROOT:?submit with ROOT}"
: "${ARCHIVE:?submit with ARCHIVE}"
: "${ACCEPTOR:?submit with ACCEPTOR}"
for path in "$ROOT" "$ARCHIVE" "$ACCEPTOR"; do
  case "$(realpath "$path")" in
    /work/home/acf9jvb3sm/*) ;;
    *) exit 2 ;;
  esac
done

EXPECTED_ARCHIVE_SHA=8bcfec3155e0eeb8ec03e859dae1a16cb1346ccce880943849f1facb87a34894
ACTUAL_ARCHIVE_SHA="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
test "$ACTUAL_ARCHIVE_SHA" = "$EXPECTED_ARCHIVE_SHA"

mkdir -p "$ROOT"
STAGING="$ROOT/cache-staging-${SLURM_JOB_ID}"
DEST="$ROOT/cache"
test ! -e "$STAGING"
test ! -e "$DEST"
mkdir -p "$STAGING"
tar -xzf "$ARCHIVE" -C "$STAGING"

PYTHON=/public/software/apps/DeepLearning/PyTorch/torch110_dtk2210_py39/bin/python
"$PYTHON" "$ACCEPTOR" "$STAGING/cache" \
  --expected-source-commit e083bee19ee6a13cd9f72e91229752a9d5f56389 \
  --output "$ROOT/cache_acceptance.json"
mv "$STAGING/cache" "$DEST"
rmdir "$STAGING"
