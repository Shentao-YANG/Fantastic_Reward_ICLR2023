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

mkdir -p "./damd_multiwoz/${data_folder}"
cp -r "./damd_multiwoz/data/multi-woz" "./damd_multiwoz/${data_folder}/"

mkdir -p "./damd_multiwoz/${data_folder}/embeddings"
if [ -e "./damd_multiwoz/data/embeddings_store/glove.6B.zip" ]
then
    printf "\n Use stored glove.6b.zip \n"
    cp ./damd_multiwoz/data/embeddings_store/glove.6B.zip "./damd_multiwoz/${data_folder}/embeddings/glove.6B.zip"
else
    printf "\n Download glove.6b.zip \n"
    wget -O "./damd_multiwoz/${data_folder}/embeddings/glove.6B.zip" https://nlp.stanford.edu/data/glove.6B.zip
    mkdir -p ./damd_multiwoz/data/embeddings_store
    cp "./damd_multiwoz/${data_folder}/embeddings/glove.6B.zip" ./damd_multiwoz/data/embeddings_store/glove.6B.zip
fi
unzip "./damd_multiwoz/${data_folder}/embeddings/glove.6B.zip" -d "./damd_multiwoz/${data_folder}/embeddings"
echo "400000 100" | cat - "./damd_multiwoz/${data_folder}/embeddings/glove.6B.100d.txt" > "./damd_multiwoz/${data_folder}/embeddings/glove.6B.100d.w2v.txt"
rm -rf "./damd_multiwoz/${data_folder}/embeddings/glove.6B.50d.txt" "./damd_multiwoz/${data_folder}/embeddings/glove.6B.100d.txt" "./damd_multiwoz/${data_folder}/embeddings/glove.6B.200d.txt" "./damd_multiwoz/${data_folder}/embeddings/glove.6B.300d.txt" "./damd_multiwoz/${data_folder}/embeddings/glove.6B.zip"
mkdir -p "./damd_multiwoz/${data_folder}/multi-woz-oppe/reward"

python -m spacy download en_core_web_sm

./setup.sh --data_folder $data_folder --test_mode $test_mode --exp_idx $exp_idx