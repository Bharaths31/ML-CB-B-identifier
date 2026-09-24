#!/usr/bin/env bash
# Quarter-data ablation sweep. Same seed, distinct --run-tag per run.
#
#   bash scripts/run_ablations.sh              # run all
#   bash scripts/run_ablations.sh --dry-run    # print commands only
#
# After it finishes, compare with:
#   python scripts/view_logs.py list
#   python scripts/diagnose_model.py --checkpoint <pt> --split test
#
# NOTE: the hard-pair *batch sampler* is not yet implemented; --hard-pairs only
# up-weights SupCon negatives (which still helps the confused pairs).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

run () {
  local tag="$1"; shift
  echo "==================================================================="
  echo "  ABLATION $tag : $*"
  echo "==================================================================="
  if [ "$DRY" = "1" ]; then
    echo "$PY local_train.py --quarter-data --run-tag $tag $*"
  else
    "$PY" local_train.py --quarter-data --run-tag "$tag" "$@"
  fi
}

# R0 baseline: no augmentation, sampler only, no mixing
run R0 --no-augment
# R1: light augmentation preset (flip + mild RRC + mild colour jitter)
run R1 --augment-preset light
# R2: R1 + cosine/ArcFace breed heads
run R2 --augment-preset light --cosine-head
# R3: R2 + breed trait heads (fill data/breed_traits.json first)
run R3 --augment-preset light --cosine-head --trait-weight 0.1
# R4: R3 + breed-aware augmentation (coat-colour breeds skip jitter)
run R4 --augment-preset light --cosine-head --breed-aug --trait-weight 0.1
# R5: R4 + confusion-driven hard pairs (run scripts/mine_confusions.py first)
run R5 --augment-preset light --cosine-head --breed-aug --trait-weight 0.1 \
       --hard-pairs outputs/metrics/confusion_pairs.json
# R6: best config + pad-to-square (keeps full-body side profiles)
run R6 --augment-preset light --cosine-head --breed-aug --trait-weight 0.1 \
       --hard-pairs outputs/metrics/confusion_pairs.json --pad-to-square
# R7: best config + group-aware (dedup) splits
run R7 --augment-preset light --cosine-head --breed-aug --trait-weight 0.1 \
       --hard-pairs outputs/metrics/confusion_pairs.json --dedup-splits

echo
echo "Done. Compare runs:  python scripts/view_logs.py list"
echo "Then diagnose each checkpoint on the test split."
