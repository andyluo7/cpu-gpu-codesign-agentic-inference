#!/bin/bash
# Phase 3: GPU utilization monitor
# Samples GPU usage at 100ms intervals and saves to CSV
# Run alongside benchmarks to correlate GPU activity with request timing

OUTPUT_DIR="${1:-/work/results/cpu-gpu-codesign/phase3}"
INTERVAL="${2:-0.1}"  # 100ms default
DURATION="${3:-300}"  # 5 min default
GPUS="${4:-0,1}"      # GPU indices

mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/gpu_utilization_$(date +%Y%m%d_%H%M%S).csv"

echo "timestamp_ms,gpu_id,gpu_use_pct,mem_use_pct,vram_used_mb,vram_total_mb,gpu_temp_c,power_w" > "$OUTPUT_FILE"

echo "GPU monitor started: interval=${INTERVAL}s, duration=${DURATION}s, gpus=${GPUS}"
echo "Output: $OUTPUT_FILE"

START=$(date +%s%N)
END_NS=$(( $(date +%s) + DURATION ))

while [ $(date +%s) -lt $END_NS ]; do
    TIMESTAMP_MS=$(( ($(date +%s%N) - START) / 1000000 ))

    for GPU_ID in $(echo "$GPUS" | tr ',' ' '); do
        # rocm-smi JSON output for structured parsing
        GPU_USE=$(rocm-smi -d $GPU_ID --showuse --json 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    card = list(d.values())[0] if d else {}
    print(card.get('GPU use (%)', '0'))
except: print('0')
" 2>/dev/null || echo "0")

        MEM_INFO=$(rocm-smi -d $GPU_ID --showmeminfo vram --json 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    card = list(d.values())[0] if d else {}
    used = int(card.get('VRAM Total Used Memory (B)', 0)) / 1048576
    total = int(card.get('VRAM Total Memory (B)', 0)) / 1048576
    pct = (used / total * 100) if total > 0 else 0
    print(f'{pct:.1f},{used:.0f},{total:.0f}')
except: print('0,0,0')
" 2>/dev/null || echo "0,0,0")

        TEMP=$(rocm-smi -d $GPU_ID --showtemp --json 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    card = list(d.values())[0] if d else {}
    print(card.get('Temperature (Sensor edge) (C)', '0'))
except: print('0')
" 2>/dev/null || echo "0")

        POWER=$(rocm-smi -d $GPU_ID --showpower --json 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    card = list(d.values())[0] if d else {}
    # Try different power field names
    for k in ['Average Graphics Package Power (W)', 'Current Socket Graphics Package Power (W)']:
        if k in card:
            print(card[k])
            break
    else:
        print('0')
except: print('0')
" 2>/dev/null || echo "0")

        echo "${TIMESTAMP_MS},${GPU_ID},${GPU_USE},${MEM_INFO},${TEMP},${POWER}" >> "$OUTPUT_FILE"
    done

    sleep "$INTERVAL"
done

echo "GPU monitor finished. $(wc -l < "$OUTPUT_FILE") samples collected."
echo "Output: $OUTPUT_FILE"
