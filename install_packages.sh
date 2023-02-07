#!/bin/bash

# create a "fantastic_reward" env in conda first
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge daal=2021.4.0 ruamel.yaml cudatoolkit=11.1 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install --upgrade --no-cache-dir pytest spyder pyqt5 pyqtwebengine pathlib scipy pip setuptools wheel numpy packaging pathlib
pip install --no-cache-dir 'spacy[cuda111]'
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
pip uninstall spacy-transformers
pip install nvidia-ml-py3 --no-cache-dir
pip install tensorflow==2.5.0 --no-cache-dir
pip install tensorflow-gpu==2.5.0 --no-cache-dir
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install transformers==2.8.0 --no-cache-dir
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
pip install pydevd --no-cache-dir
pip install --upgrade --no-cache-dir gensim pandas typing-extensions traitlets nest-asyncio jupyter-core ipykernel jupyter-client parso jedi pluggy click

python -c "import tensorflow; print(f'tf: {tensorflow.__version__}')"
python -c "from tensorflow import keras; print(f'keras: {keras.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import torch; print(f'pytorch: {torch.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
