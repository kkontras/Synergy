
# for fold in 0; do
#     for l in 0 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 ; do
#       python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --frozen --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
#done
#done
s
l_values=(0 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100)
results=()
for l in "${l_values[@]}"; do
  read acc std < <(
    python show_v2.py \
      --config ./configs/CREMA_D/synergy/nov/synprom_ib.json \
      --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json \
      --pre --frozen --l "$l" --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --no-printing
  )
  results+=("l=$l acc=$acc std=$std")
  echo "l=$l → acc=$acc ± $std"
done

accs=()
stds=()

for r in "${results[@]}"; do
  acc=$(echo "$r" | awk -F'acc=' '{print $2}' | awk '{print $1}')
  std=$(echo "$r" | awk -F'std=' '{print $2}')
  accs+=("$acc")
  stds+=("$std")
done

echo
echo "lambdas_no_contrastive = np.array([$(IFS=, ; echo "${l_values[*]}")])"
echo "perf_no_contrastive = np.array([$(IFS=, ; echo "${accs[*]}")])"
echo "std_no_contrastive = np.array([$(IFS=, ; echo "${stds[*]}")])"
