for f in processed/graphs/*.pt; do
    python 06_predict.py \
      --graph_path "$f" \
      --checkpoint checkpoints/best_model.pt \
      --output_dir predictions \
      --device cpu
done
