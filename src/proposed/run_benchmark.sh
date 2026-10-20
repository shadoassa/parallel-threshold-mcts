#!/bin/bash
# 使用法: ./run_benchmark.sh [prob_file] [itr_num] [delta]
# 例:     ./run_benchmark.sh prob_randomtree/prob_0.txt 20 1e-1
#
# 1, 2, 4 コアでそれぞれ実行し, 停止時刻の平均・標準偏差と壁時計時間を出力する.

PROB="${1:-prob_randomtree/prob_0.txt}"
ITR="${2:-20}"
DELTA="${3:-1e-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "./experiment_mpi" ]; then
  echo "[build] experiment_mpi not found, running make..."
  make || { echo "Build failed."; exit 1; }
fi

if [ ! -f "$PROB" ]; then
  echo "[gen] Problem file not found: $PROB"
  echo "[gen] Generating with gen_prob_review.py..."
  python3 gen_prob_review.py || { echo "gen_prob_review.py failed."; exit 1; }
fi

echo "=== Benchmark ==="
echo "  prob:  $PROB"
echo "  itr:   $ITR"
echo "  delta: $DELTA"
echo ""

for NP in 1 2 4; do
  OUTFILE="result_np${NP}.txt"
  echo "--- np=$NP ---"

  START_NS=$(date +%s%N)
  mpirun --oversubscribe -np "$NP" ./experiment_mpi \
    -p "$PROB" -d "$DELTA" -i "$ITR" -o "$OUTFILE" --silent
  END_NS=$(date +%s%N)

  WALL_MS=$(( (END_NS - START_NS) / 1000000 ))
  WALL_SEC=$(echo "scale=3; $WALL_MS / 1000" | bc)

  echo "  wall time : ${WALL_SEC} sec"
  grep -E "tau_avg|tau_stddev|correct_count" "$OUTFILE" | sed 's/^/  /'
  echo ""
done
