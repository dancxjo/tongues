#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server_url="${TONGUES_SERVER_URL:-http://127.0.0.1:3000}"
session_id="run:phonetic-v1"
fixture="$repo_root/fixtures/timeline/phonetic-segmentation-inspection-v1.json"

response="$(
  curl --fail --silent --show-error \
    -X PUT \
    -H 'Content-Type: application/json' \
    --data-binary "@$fixture" \
    "$server_url/api/timeline/sessions/$session_id"
)"

jq -e '
  .session.session_id == "run:phonetic-v1"
  and .session.attachments[0].kind == "phonetic_segmentation"
  and .session.attachments[0].payload.readiness == "partial"
  and ([.session.evidence[].modality] | contains(["audio", "transcript", "word", "phone", "phoneme", "speaker"]))
' <<<"$response" >/dev/null

phone_id='phonetic-segmentation:phones-v1:3'
query="span=$phone_id&start_ms=370&end_ms=505"
printf 'Fixture session saved and verified.\n'
printf 'Inspect: %s/runs/%s/tracks?%s\n' "$server_url" "$session_id" "$query"
printf 'Correct: %s/sessions/%s/correct?%s\n' "$server_url" "$session_id" "$query"
