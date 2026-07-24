#!/usr/bin/env bash
# Move validated monolithic 2D caches to the persistent 400 GB work1 share.
#SBATCH --job-name=molgap-archive-2d-graphs
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --output=results/scnet/logs/%x-%j.out
#SBATCH --error=results/scnet/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
DEST_ROOT="/work1/share/acf9jvb3sm/molgap-cache/phase8/multi2d_graphs"
mkdir -p "$DEST_ROOT" results/scnet/logs

archive_one() {
  local source="$1"
  local name="$2"
  local expected_bytes="$3"
  local destination="$DEST_ROOT/$name"
  local temporary="$DEST_ROOT/.$name.tmp"
  local resolved_source
  resolved_source="$(readlink -f "$source")"
  if [[ "$resolved_source" == "$destination" ]]; then
    echo "already_archived source=$source destination=$destination"
    return
  fi
  [[ -f "$source" ]] || { echo "missing source: $source" >&2; return 1; }
  [[ "$(stat -c %s "$source")" == "$expected_bytes" ]] || {
    echo "unexpected source size: $source" >&2
    return 1
  }
  if [[ ! -f "$destination" || "$(stat -c %s "$destination")" != "$expected_bytes" ]]; then
    rm -f "$temporary"
    cp "$source" "$temporary"
    [[ "$(stat -c %s "$temporary")" == "$expected_bytes" ]]
    mv -f "$temporary" "$destination"
  fi
  local source_sha destination_sha
  source_sha="$(sha256sum "$source" | awk '{print $1}')"
  destination_sha="$(sha256sum "$destination" | awk '{print $1}')"
  [[ "$source_sha" == "$destination_sha" ]] || {
    echo "SHA-256 mismatch for $name" >&2
    return 1
  }
  rm -f "$source"
  ln -s "$destination" "$source"
  [[ "$(readlink -f "$source")" == "$destination" ]]
  printf '%s|%s|%s|%s\n' "$name" "$expected_bytes" "$source_sha" "$destination"
}

MANIFEST_TMP="$DEST_ROOT/.SHA256_MANIFEST.tmp"
{
  archive_one \
    results/phase8/multi2d_2m_scnet/pyg_2d_graphs_multi2d_coverage_2m.pt \
    pyg_2d_graphs_multi2d_coverage_2m.pt 9206943150
  archive_one \
    results/phase8/multi2d_2m_hard20k_scnet/pyg_2d_graphs_multi2d_2m_hard20k.pt \
    pyg_2d_graphs_multi2d_2m_hard20k.pt 9329592234
} > "$MANIFEST_TMP"
mv -f "$MANIFEST_TMP" "$DEST_ROOT/SHA256_MANIFEST"
df -h /public/home/acf9jvb3sm /work1/share/acf9jvb3sm
cat "$DEST_ROOT/SHA256_MANIFEST"
