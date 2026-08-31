while read base grad; do
    python 09_flexoelectric_coefficient.py \
      --polarization_csv "polarization/${base}_polarization.csv" \
      --strain_gradient "$grad" \
      --component z \
      --prefix "$base"
done < strain_gradients.txt
