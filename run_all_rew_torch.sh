# /usr/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
    	--SEED) SEED="$2"; shift ;;
      --CUDA_DEVICE) CUDA_DEVICE="$2"; shift ;;
      --EXP_IDX) EXP_IDX="$2"; shift ;;
      --REWARD_SAMPLES) REWARD_SAMPLES="$2"; shift ;;         # default: 2
      --REWARD_LOSS) REWARD_LOSS="$2"; shift ;;               # default: "listNet"
      --LISTMLE_TEMP) LISTMLE_TEMP="$2"; shift ;;             # default: "1"
      --LISTNET_POW) LISTNET_POW="$2"; shift ;;               # default: 1
      --POLICY_TRAIN_DATA_FRAC) POLICY_TRAIN_DATA_FRAC="$2"; shift ;;               # default: 1.
      --NEG_REW_WEIGHT) NEG_REW_WEIGHT="$2"; shift ;;             # default: 0 (do not maximize reward when training the policy)
      --REW_MODEL_EXP) REW_MODEL_EXP="$2"; shift ;;             # default: 0 (exp_idx to copy reward model from, "0" do not copy)
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
#################################################
K=10
GAMMA=0.0
TEST_MODE=0
BATCH_SIZE=4
#################################################
NUM_THREAD=4
#################################################
STORE_LOC="./python_outputs/Exp${EXP_IDX}"
mkdir -p "${STORE_LOC}"
EXP_ID="s${SEED}_K${K}_GAMMA${GAMMA}"
DATA_FOLDER="Exp${EXP_IDX}data/${EXP_ID}"
OUTPUT_FILE="${STORE_LOC}/${EXP_ID}.txt"
printf "\nUse RewardTorch with BART !!!\n"
printf "\n [Exp ${EXP_IDX}] seed=${SEED}, K=${K}, GAMMA=${GAMMA}, CUDA_DEVICE=${CUDA_DEVICE}, NUM_THREAD=${NUM_THREAD}, OUTPUT_FILE=${OUTPUT_FILE}, EXP_ID=${EXP_ID}, TEST_MODE=${TEST_MODE} \n"
printf " [Exp ${EXP_IDX}] REWARD_SAMPLES=${REWARD_SAMPLES}, REW_MODEL_EXP=${REW_MODEL_EXP} \n"
#################################################
CURR_DATA_PATH="./damd_multiwoz/Exp${EXP_IDX}data/${EXP_ID}/"
if [ ${REW_MODEL_EXP} == '0' ]; then
  printf "\n Do not copy reward model, proceed with reward model training !!!\n"
  STORE_DATA_PATH="./damd_multiwoz/ExpStoreddata/s111_K${K}_GAMMA${GAMMA}/."
  mkdir -p "${CURR_DATA_PATH}"
  cp -r "${STORE_DATA_PATH}" "${CURR_DATA_PATH}"
  printf "Copied file from ${STORE_DATA_PATH} to ${CURR_DATA_PATH} !!! \n"

  printf "\n [Exp${EXP_IDX}|${EXP_ID}] Reward Learning at  $(date) \n"
  OMP_NUM_THREADS=$NUM_THREAD MKL_NUM_THREADS=$NUM_THREAD CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python RewardTorch.py \
      --data_folder ${DATA_FOLDER} --exp_idx ${EXP_IDX} --folds ${K} --reward_learning_samples ${REWARD_SAMPLES} \
      --reward_loss ${REWARD_LOSS} --listmle_temp ${LISTMLE_TEMP} --listnet_power ${LISTNET_POW} --policy_training_seed ${SEED} \
      --cfg seed=11 cuda_device=0 batch_size=${BATCH_SIZE} early_stop_count=7 >> ${OUTPUT_FILE}

  printf "\n [Exp${EXP_IDX}|${EXP_ID}] Estimate Behavior Policy at $(date) \n"
  OMP_NUM_THREADS=$NUM_THREAD MKL_NUM_THREADS=$NUM_THREAD CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python EstimateBehaviorPolicy.py \
      --seed 111 --folds $K --action_space resp --gamma ${GAMMA} --metric soft --data_folder ${DATA_FOLDER} >> ${OUTPUT_FILE}
else
  printf "\n Copied reward model from Exp${REW_MODEL_EXP}, skip reward model training !!!\n"
fi

printf "\n [Exp${EXP_IDX}|${EXP_ID}] Training response generation model at $(date) \n"
OMP_NUM_THREADS=$NUM_THREAD MKL_NUM_THREADS=$NUM_THREAD CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py \
                                    --mode train --context_window 2 --pretrained_checkpoint bart-large-cnn \
                                    --gradient_accumulation_steps 8 --lr 3e-5 --back_bone bart \
                                    --cfg seed=${SEED} cuda_device=0 batch_size=8 early_stop_count=7 \
                                    --caspi_returns_file="fn_Gs_${K}_${GAMMA}_resp_soft.json" --caspi_wt=5. \
                                    --caspi_data_file=data_for_damd.json --caspi_val_fraction=.5 --caspi --data_folder ${DATA_FOLDER} \
                                    --exp_idx ${EXP_IDX} --fraction ${POLICY_TRAIN_DATA_FRAC} \
                                    --neg_rew_weight ${NEG_REW_WEIGHT} >> ${OUTPUT_FILE}
sleep 10s
printf "\n [Exp${EXP_IDX}|${EXP_ID}] Finished at $(date) \n"

