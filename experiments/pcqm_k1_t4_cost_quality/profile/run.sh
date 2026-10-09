#!/usr/bin/env bash
set -euo pipefail
# Parent supplies verified payload and optional frozen setup script. No submission.
root=$1
output=$2
manifest_sha=$3
setup=${4:-}
remaining=${5:-1200}
[[ "$remaining" =~ ^[0-9]+$ ]] && (( remaining > 0 && remaining <= 1200 ))
exec timeout --signal=KILL "${remaining}s" bash -c '
set -euo pipefail
if [ -n "$4" ]; then bash "$4"; fi
export PYTHONPATH="$1/src"
exec python -m molgap.k1_execution_profile --native-t4 --root "$1" --output "$2" --expected-manifest-sha256 "$3"
' profile "$root" "$output" "$manifest_sha" "$setup"
