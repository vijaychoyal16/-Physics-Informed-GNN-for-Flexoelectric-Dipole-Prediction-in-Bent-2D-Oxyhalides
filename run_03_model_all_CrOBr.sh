#!/bin/bash

# ============================================================
# Run 03_model.py test for flat CrOBr and bent CrOBr_0.1 to CrOBr_5.0
#
# Required graph files:
#   processed/graphs/CrOBr.pt
#   processed/graphs/CrOBr_0.1.pt
#   ...
#   processed/graphs/CrOBr_5.0.pt
# ============================================================

DEVICE="cpu"
GRAPH_DIR="processed/graphs"

echo "============================================================"
echo "Testing flat CrOBr graph"
echo "============================================================"

if [ -f "${GRAPH_DIR}/CrOBr.pt" ]; then
    python 03_model.py \
        --graph_path "${GRAPH_DIR}/CrOBr.pt" \
        --device "$DEVICE"
else
    echo "WARNING: Missing ${GRAPH_DIR}/CrOBr.pt"
fi

echo
echo "============================================================"
echo "Testing bent CrOBr graphs from 0.1 to 5.0"
echo "============================================================"

for n in $(seq 1 50); do

    A=$(awk -v n="$n" 'BEGIN {printf "%.1f", n*0.1}')

    GRAPH_FILE="${GRAPH_DIR}/CrOBr_${A}.pt"

    echo
    echo "------------------------------------------------------------"
    echo "Testing graph: $GRAPH_FILE"
    echo "------------------------------------------------------------"

    if [ -f "$GRAPH_FILE" ]; then
        python 03_model.py \
            --graph_path "$GRAPH_FILE" \
            --device "$DEVICE"
    else
        echo "WARNING: Missing $GRAPH_FILE"
    fi

done

echo
echo "Done testing all available graphs."
