for f in predictions/*_total_prediction.csv; do
    base=$(basename "$f" _total_prediction.csv)

    python 08_calculate_polarization.py \
      --total_csv "predictions/${base}_total_prediction.csv" \
      --atomic_csv "predictions/${base}_atomic_dipoles.csv" \
      --prefix "$base"
done
