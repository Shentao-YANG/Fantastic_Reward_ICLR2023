# Fantastic Rewards and How to Tame Them: A Case Study on Reward Learning for Task-Oriented Dialogue Systems 

***

## Dependency
To install the required packages, first create and activate a `fantastic_reward` env in conda.
Then execute the following command:
```angular2html
bash install_packages.sh
```

***
## Experiments

### Data Setup
Our data-setup follows the [**CASPI**](https://github.com/salesforce/CASPI) paper.
Please download the pre-processed data from [here](https://drive.google.com/file/d/15A88j-pyI-jBznKmvJ17HtgW1h7fwDEl/view?usp=sharing).
Unzip the downloaded file and put the resulting folder `ExpStoreddata` into the folder `damd_multiwoz`.

### Training the Reward and Response Models

For our variant of RewardNet+GS $N = 3,\Phi=(\cdot)^1$ in Table 1 of the paper, please run the following command
```angular2html
bash ./run_multiple_seeds.sh --EXP_IDX ${EXP_IDX} --REWARD_SAMPLES 3 --REWARD_LOSS "listNet" --LISTMLE_TEMP 1 --LISTNET_POW 1 --POLICY_TRAIN_DATA_FRAC 1 --NEG_REW_WEIGHT 0.1 --REW_MODEL_EXP '0'
```
where `${EXP_IDX}` is the index of the experiment, such as `"2023"`.

For our variant of RewardMLE+GS $N = 5,\Phi=\exp(\cdot)$ in Table 1 of the paper, please run the following command
```angular2html
bash ./run_multiple_seeds.sh --EXP_IDX ${EXP_IDX} --REWARD_SAMPLES 5 --REWARD_LOSS "listMLE" --LISTMLE_TEMP 1 --LISTNET_POW 0 --POLICY_TRAIN_DATA_FRAC 1 --NEG_REW_WEIGHT 1.0 --REW_MODEL_EXP '0'
```
where `${EXP_IDX}` is again the index of the experiment.

### Evaluating the Released Checkpoints

To facilitate reproducibility, we release a checkpoint for each of the variant 
RewardNet+GS $N = 3,\Phi=(\cdot)^1$ and RewardMLE+GS $N = 5,\Phi=\exp(\cdot)$ in Table 1 of the paper.
The released checkpoints are both trained under random seed `999` of the tested five seeds `(111 333 555 777 999)`.

To evaluate the checkpoints, please try the following steps.
Here `Exp1` corresponds to the variant of RewardNet+GS $N =3,\Phi=(\cdot)^1$ and `Exp2` for RewardMLE+GS $N = 5,\Phi=\exp(\cdot)$.

1. Download and unzip the checkpoints from [here](https://drive.google.com/file/d/1EUIno8hq94smUqBBnzr_m8svMWKKH7P5/view?usp=sharing). Put the resulting folders into a  folder named `experiments`.
2. Download and unzip the processed data from [here](https://drive.google.com/file/d/1fwLK62U38B3pxYxzrycGyEwt4_AFRv7l/view?usp=sharing). Put the resulting folders into the folder `damd_multiwoz`.
3. Try the following command
```angular2html
python train.py --model_path "Exp${EXP_IDX}/all_sd999/" \
    --mode 'test' --context_window 2 --pretrained_checkpoint bart-large-cnn \
    --back_bone bart --cfg seed=999 cuda_device=0 batch_size=8 early_stop_count=7 \
    --caspi_returns_file="fn_Gs_10_0.0_resp_soft.json" --caspi_wt=5. \
    --caspi_data_file=data_for_damd.json --caspi_val_fraction=.5 --caspi --data_folder "Exp${EXP_IDX}data/s999_K10_GAMMA0.0" \
    --exp_idx ${EXP_IDX} 
```
where `${EXP_IDX}` should be replaced by `1` or `2`.


## Acknowledgement
This codebase builds on the following codebases and datasets:
* [**CASPI**](https://github.com/salesforce/CASPI).
* [**MinTL**](https://github.com/zlinao/MinTL).
* [**DAMD**](https://gitlab.com/ucdavisnlp/damd-multiwoz).
* [**Multiwoz2.0**](https://github.com/budzianowski/multiwoz).
* [**ConvLab Multiwoz2.0 annotation**](https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz/annotation).