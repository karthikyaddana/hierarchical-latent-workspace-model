#!/usr/bin/env bash
# Mint the two per-paper Zenodo DOIs.
#
# Prompts for the token without echoing it, keeps it in the process environment only,
# and never writes it to disk or to shell history.
#
#   ./tools/mint_paper_dois.sh
#
# Creates DRAFT deposits. Nothing becomes public until you press Publish in the Zenodo
# UI. Token needs the deposit:write and deposit:actions scopes.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${ZENODO_TOKEN:-}" ]]; then
  printf 'Zenodo token (input hidden): ' >&2
  IFS= read -rs ZENODO_TOKEN || true
  printf '\n' >&2
fi
export ZENODO_TOKEN

if [[ -z "$ZENODO_TOKEN" ]]; then
  echo "error: no token entered" >&2
  exit 1
fi

for target in postmortem hlwm-paper; do
  echo
  echo "=============================================================="
  echo " $target"
  echo "=============================================================="
  python3 tools/zenodo_deposit.py --target "$target" --apply
done

echo
echo "Both drafts created. Review them at https://zenodo.org/me/uploads"
echo "and press Publish on each to mint the DOIs."
echo
echo "Then rotate this token:"
echo "  https://zenodo.org/account/settings/applications/tokens/"
