#!/bin/bash

OUT="strain_gradients.csv"

echo "id,strain_gradient" > "$OUT"
echo "flat_CrOBr,0.0" >> "$OUT"

START=0.1
STEP=0.1
NPOINTS=33
SCALE=2.0e6

for n in $(seq 1 "$NPOINTS"); do
    A=$(awk -v start="$START" -v step="$STEP" -v n="$n" 'BEGIN {printf "%.1f", start + (n-1)*step}')
    GRAD=$(awk -v A="$A" -v scale="$SCALE" 'BEGIN {printf "%.6e", A*scale}')
    echo "CrOBr_${A},${GRAD}" >> "$OUT"
done

echo "Created $OUT"
cat "$OUT"
