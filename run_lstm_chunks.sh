#!/bin/bash

#Created with claude
# ── Configuration ────────────────────────────────────────────────────────────
TOTAL_CHUNKS=2 #20         # one per core, adjust to available cores
RAM_PER_CHUNK=4 #in GB
PYTHON="python3"        # adjust if needed, e.g. python3.11
SCRIPT="main.py"
LOG_DIR="./logs/lstm_chunks"
 
mkdir -p "$LOG_DIR"
mkdir -p "./results/LSTM"
 
echo "=========================================="
echo "Starting $TOTAL_CHUNKS LSTM chunks"
echo "Ram limit per chunk: $RAM_PER_CHUNK GB"
echo "Logs: $LOG_DIR"
echo "=========================================="
 
# Launch all chunks in parallel, each as independent process
for i in $(seq 0 $((TOTAL_CHUNKS - 1))); do
    echo "Launching chunk $i / $((TOTAL_CHUNKS - 1))..."
    $PYTHON "$SCRIPT" \
        --model lstm \
        --chunk "$i" \
        --total_chunks "$TOTAL_CHUNKS" \
        > "$LOG_DIR/chunk_${i}.log" 2>&1 &
done
 
echo "All $TOTAL_CHUNKS chunks launched. Waiting for completion..."
wait
echo ""
echo "=========================================="
echo "All chunks finished. Merging results..."
echo "=========================================="
 
$PYTHON merge_lstm_results.py
 
echo "Done."
 