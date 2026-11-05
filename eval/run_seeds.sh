#!/bin/bash
# Robustness check: regenerate EVERY seeded dataset with shifted random seeds and
# replay the full evaluation for each, so the fault rules can't be tuned to one
# lucky dataset. Runs on a throwaway copy of the committed tree; the repo is untouched.
#   usage: eval/run_seeds.sh [seed ...]      (default: 11 22 33 44 55 66 77 88 99 111)
set -u
SEEDS=${*:-"11 22 33 44 55 66 77 88 99 111"}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
for seed in $SEEDS; do
  d="$WORK/s$seed"; mkdir -p "$d"
  git -C "$ROOT" archive HEAD | tar -x -C "$d"
  for f in "$d"/data/generate.py "$d"/data/generate_bench.py "$d"/data/gen_*.py; do
    perl -pi -e "s/default_rng\((\d+)\)/'default_rng(' . (\$1 + $seed) . ')'/ge" "$f"
  done
  (cd "$d" && python data/generate.py >/dev/null 2>&1)
  result=$(cd "$d" && SANDBOX_MODE=local RECEIPT_LEDGER="$d/r.jsonl" LEDGER_KEY_FILE="$d/k" \
           LEDGER_SIGNING_KEY_FILE="$d/k.pem" RESULTS_DIR="$d/results" python eval/run_eval.py | grep -E 'FAIL|correct')
  echo "seed +$seed: $(echo "$result" | tr '\n' ' ')"
done
