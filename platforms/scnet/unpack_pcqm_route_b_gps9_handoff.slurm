#!/bin/bash
#SBATCH --job-name=rb-gps9-unpack
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --gres=dcu:1
#SBATCH --time=01:00:00

set -euo pipefail

: "${ARCHIVE:?submit with ARCHIVE}"
: "${DEST:?submit with DEST}"
: "${EXPECTED_SHA:?submit with EXPECTED_SHA}"

home_root="$(realpath "$HOME")"
archive_path="$(realpath "$ARCHIVE")"
case "$archive_path" in
  "$home_root"/*) ;;
  *) echo "archive escapes user home: $archive_path" >&2; exit 1 ;;
esac

mkdir -p "$DEST"
dest_path="$(realpath "$DEST")"
share_root="/work1/share/$USER"
case "$dest_path" in
  "$share_root"/*) ;;
  *) echo "destination escapes user share: $dest_path" >&2; exit 1 ;;
esac

actual_sha="$(sha256sum "$archive_path" | awk '{print $1}')"
test "$actual_sha" = "$EXPECTED_SHA"
printf "%s  %s\n" "$actual_sha" "$(basename "$archive_path")" \
  > "${archive_path}.sha256"

if tar -tf "$archive_path" | awk \
  '/^(\/|\.\.\/)|\/\.\.\// {bad=1; print} END{exit bad}'; then
  echo "tar member paths are safe"
else
  exit 1
fi

tar -xf "$archive_path" -C "$dest_path"
cd "$dest_path"
sha256sum -c gps9_resume_handoff_20260729.sha256
test "$(find inputs/pcqm_route_b_1m/gps -maxdepth 1 -type f -name '*.pt' | wc -l)" -eq 401
test -f outputs_tuned/gps9/last.pt
test -f warmstarts/gps9.pt
test -f search_winners/gps9.json

cat > handoff_acceptance.txt <<EOF
status=accepted
archive_sha256=$actual_sha
archive_bytes=$(stat -c %s "$archive_path")
gps_graph_parts=401
resume_checkpoint_bytes=$(stat -c %s outputs_tuned/gps9/last.pt)
source_assets_deleted=false
EOF
cat handoff_acceptance.txt
