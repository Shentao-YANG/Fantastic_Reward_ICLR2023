#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
    	--data_folder) data_folder="$2"; shift ;;
      --test_mode) test_mode="$2"; shift ;;
      --exp_idx) exp_idx="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cd damd_multiwoz
python -m spacy download en_core_web_sm
# preprocessing
python data_analysis.py --data_folder $data_folder
python preprocess.py --data_folder $data_folder --test_mode $test_mode --exp_idx $exp_idx
# setup python path
# type pwd inside damd_multiwoz to find out the path of damd_multiwoz folder
export PYTHONPATH=`pwd`
cd ..

