# /usr/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
      --EXP_IDX) EXP_IDX="$2"; shift ;;                       # default: 1
      --REWARD_SAMPLES) REWARD_SAMPLES="$2"; shift ;;         # default: 2
      --REWARD_LOSS) REWARD_LOSS="$2"; shift ;;               # default: "listNet"
      --LISTMLE_TEMP) LISTMLE_TEMP="$2"; shift ;;             # default: 1
      --LISTNET_POW) LISTNET_POW="$2"; shift ;;               # default: 1
      --POLICY_TRAIN_DATA_FRAC) POLICY_TRAIN_DATA_FRAC="$2"; shift ;;               # default: 1.
      --NEG_REW_WEIGHT) NEG_REW_WEIGHT="$2"; shift ;;             # default: 0 (do not maximize reward when training the policy)
      --REW_MODEL_EXP) REW_MODEL_EXP="$2"; shift ;;             # default: 0 (exp_idx to copy reward model from, "0" do not copy)
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
###############################################
ALL_SEEDS=(111 333 555 777 999)
ALL_DEVICES=(0 1 2 3 4)
#################################################
printf "\n Begin at $(date) \n"

if [ -e "./damd_multiwoz/data/embeddings_store/glove.6B.zip" ]; then
  printf "\n Use stored glove.6b.zip \n"
else
  printf "\n Download glove.6b.zip once \n"
  mkdir -p ./damd_multiwoz/data/embeddings_store
  wget -O "./damd_multiwoz/data/embeddings_store/glove.6B.zip" https://nlp.stanford.edu/data/glove.6B.zip
fi

if [ ${REW_MODEL_EXP} == '0' ]; then
  printf "\n Do not copy reward model \n"
else
  printf "\n Copy reward model from Exp${REW_MODEL_EXP} \n"
  mkdir -p "./damd_multiwoz/Exp${EXP_IDX}data/"
  cp -r "./damd_multiwoz/Exp${REW_MODEL_EXP}data/." "./damd_multiwoz/Exp${EXP_IDX}data/"
  for seed in "${ALL_SEEDS[@]}"; do
    mkdir -p "./experiments/Exp${EXP_IDX}/all_sd${seed}/"
    cp -r "./experiments/Exp${REW_MODEL_EXP}/all_sd${seed}/reward_model" "./experiments/Exp${EXP_IDX}/all_sd${seed}/"
  done
  wait
fi


for ((index=0; index<"${#ALL_SEEDS[@]}"; index+=1)); do
  SEED=${ALL_SEEDS[$index]}
  CUDA_DEVICE=${ALL_DEVICES[$index]}
  bash ./run_all_rew_torch.sh --SEED ${SEED} --CUDA_DEVICE ${CUDA_DEVICE} --EXP_IDX ${EXP_IDX} --REWARD_SAMPLES ${REWARD_SAMPLES} \
    --REWARD_LOSS ${REWARD_LOSS} --LISTMLE_TEMP ${LISTMLE_TEMP} --LISTNET_POW ${LISTNET_POW} --POLICY_TRAIN_DATA_FRAC ${POLICY_TRAIN_DATA_FRAC} \
    --NEG_REW_WEIGHT ${NEG_REW_WEIGHT} --REW_MODEL_EXP ${REW_MODEL_EXP} &
  sleep 120s
done
wait

sleep 10s
printf "\n Finish at $(date) \n"


