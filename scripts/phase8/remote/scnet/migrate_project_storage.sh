#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/molgap}"
DEST_ROOT="${DEST_ROOT:-/work1/share/$USER/molgap_storage}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 RELATIVE_PATH [RELATIVE_PATH ...]" >&2
  exit 2
fi

project_real="$(readlink -f "$PROJECT_ROOT")"
mkdir -p "$DEST_ROOT/.migration_manifests"
dest_real="$(readlink -f "$DEST_ROOT")"

for rel in "$@"; do
  if [[ "$rel" = /* || "$rel" == *".."* ]]; then
    echo "refusing unsafe relative path: $rel" >&2
    exit 2
  fi

  source_path="$PROJECT_ROOT/$rel"
  final_path="$DEST_ROOT/$rel"
  stage_path="$final_path.partial"
  partial_path="$DEST_ROOT/.rsync-partial/${rel//\//__}"
  marker="$DEST_ROOT/.migration_manifests/${rel//\//__}.done"

  if [[ -L "$source_path" ]]; then
    current_target="$(readlink -f "$source_path")"
    expected_target="$(readlink -f "$final_path")"
    if [[ "$current_target" == "$expected_target" ]]; then
      echo "already migrated: $rel -> $current_target"
      continue
    fi
    echo "refusing unexpected symlink: $source_path -> $current_target" >&2
    exit 1
  fi

  [[ -d "$source_path" ]] || { echo "missing directory: $source_path" >&2; exit 1; }
  source_real="$(readlink -f "$source_path")"
  [[ "$source_real" == "$project_real/"* ]] || {
    echo "source escapes project root: $source_real" >&2
    exit 1
  }
  [[ "$final_path" == "$dest_real/"* ]] || {
    echo "destination escapes storage root: $final_path" >&2
    exit 1
  }
  [[ ! -e "$final_path" ]] || {
    echo "destination already exists without project symlink: $final_path" >&2
    exit 1
  }

  mkdir -p "$(dirname "$stage_path")" "$stage_path" "$partial_path"
  echo "syncing: $source_path -> $stage_path"
  rsync -a --checksum --partial-dir="$partial_path" "$source_path/" "$stage_path/"

  differences="$(rsync -naci --delete "$source_path/" "$stage_path/")"
  if [[ -n "$differences" ]]; then
    echo "verification failed for $rel:" >&2
    printf '%s\n' "$differences" >&2
    exit 1
  fi

  mv "$stage_path" "$final_path"
  rm -rf -- "$source_real"
  ln -s "$final_path" "$source_path"
  rm -rf -- "$partial_path"

  marker_tmp="$marker.tmp.$$"
  {
    printf 'relative_path=%s\n' "$rel"
    printf 'destination=%s\n' "$final_path"
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$marker_tmp"
  mv "$marker_tmp" "$marker"
  echo "migrated: $rel -> $final_path"
done
