#!/usr/bin/env bash
# Upload the HLWM training corpora to a Hugging Face dataset repo.
#
# Dry run by default. Prints the plan, uploads nothing until you pass --apply.
#
#   HF_DATASET_REPO=<user>/hlwm-corpora ./tools/upload_data_hf.sh
#   HF_DATASET_REPO=<user>/hlwm-corpora ./tools/upload_data_hf.sh --apply
#
# Requires: `hf auth login` (or HF_TOKEN in the environment).

set -euo pipefail

REPO="${HF_DATASET_REPO:-}"
SRC="${HLWM_DATA_DIR:-data}"
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

if [[ -z "$REPO" ]]; then
  echo "error: set HF_DATASET_REPO, e.g. HF_DATASET_REPO=yourname/hlwm-corpora" >&2
  exit 1
fi

# reasoning9000 is excluded by default: it carries provenance records tied to ingested
# third-party source documents. Set INCLUDE_REASONING9000=1 only after a licensing review.
DIRS=(combined hlwm-v5.6 expert-beast beast final reviewed generated chunks expert_seeds)
[[ "${INCLUDE_REASONING9000:-0}" == "1" ]] && DIRS+=(reasoning9000)

echo "target dataset : $REPO"
echo "source dir     : $SRC"
echo "identity       : $(hf auth whoami 2>/dev/null | tail -1 || echo '<not logged in>')"
echo

total=0
for d in "${DIRS[@]}"; do
  if [[ -d "$SRC/$d" ]]; then
    size=$(du -sh "$SRC/$d" | cut -f1)
    printf '  %-16s %8s\n' "$d" "$size"
    total=$((total + $(du -sk "$SRC/$d" | cut -f1)))
  else
    printf '  %-16s %8s\n' "$d" "(missing)"
  fi
done
echo
printf 'total: %s MB\n' "$((total / 1024))"

if [[ $APPLY -eq 0 ]]; then
  echo
  echo "dry run — re-run with --apply to upload."
  exit 0
fi

hf repo create "$REPO" --repo-type dataset --exist-ok
for d in "${DIRS[@]}"; do
  [[ -d "$SRC/$d" ]] || continue
  echo ">>> uploading $d"
  hf upload "$REPO" "$SRC/$d" "$d" --repo-type dataset
done
echo "done: https://huggingface.co/datasets/$REPO"
