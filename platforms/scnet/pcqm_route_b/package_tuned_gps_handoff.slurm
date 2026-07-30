#!/usr/bin/env bash
#SBATCH --job-name=rb-pcqm-package
#SBATCH --partition=hx1hdnormal
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G

set -euo pipefail

: "${ENCODER_NAME:?submit with ENCODER_NAME=gps9 or gps11_160}"
: "${ROOT:?submit with ROOT}"
: "${ARCHIVE_NAME:?submit with ARCHIVE_NAME}"

root_path="$(realpath "$ROOT")"
case "$root_path" in
  "/work1/share/$USER"/*) ;;
  *) echo "ROOT escapes user share: $root_path" >&2; exit 1 ;;
esac
case "$ARCHIVE_NAME" in
  *[!A-Za-z0-9._-]* | "") echo "invalid ARCHIVE_NAME" >&2; exit 1 ;;
esac

cd "$root_path"
output_dir="outputs_tuned/$ENCODER_NAME"
test -f "$output_dir/single_encoder_acceptance.json"
test -f "$output_dir/completion_manifest.json"
test -f "search_winners/$ENCODER_NAME.json"
mkdir -p handoffs

temporary="handoffs/.${ARCHIVE_NAME}.tmp"
final="handoffs/${ARCHIVE_NAME}"
test ! -e "$temporary"
test ! -e "$final"
tar -cf "$temporary" \
  "$output_dir" \
  "search_winners/$ENCODER_NAME.json" \
  "preflight/${ENCODER_NAME}_resume_contract.json"
mv "$temporary" "$final"
sha256sum "$final" > "${final}.sha256"

cat > "${final}.manifest" <<EOF
status=complete
encoder=$ENCODER_NAME
archive=$(basename "$final")
archive_bytes=$(stat -c %s "$final")
archive_sha256=$(sha256sum "$final" | awk '{print $1}')
source_output_deleted=false
EOF
cat "${final}.manifest"
