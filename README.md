# nextvlad-for-nlp
This repository contains the code for our paper titled "An Investigation into the Contribution of Locally Aggregated Descriptors to Figurative Language Identification".

For reproducibility and more deterministic execution, run with:
<br>CUBLAS_WORKSPACE_CONFIG=:16:8 python tbinv_earlystop.py


upload the requirements list (pip freeze file)

The current version of the implementation uses a linear scheduler with number of warmups = 2000 and initial learning rate = 1e^-6. However, we also tried other learning rate strategies including cyclic LR and observed no improvements. For the cyclic version, we use base_lr = 1e^-7 and max_lr = 1e^-3.


If you want to understand our code and maybe develop based on that, we encourage you to first read our main script which is the tbinv_earlystop.py.


__tbinv_earlystop.py__
This is the main script and creates a BERT + BiLSTM + NeXtVLAD model and fine-tunes it. It also applies the early stopping strategy with patience=2 and delta=0. This script is documented to help the reader understand the goal of each part.


__ensemble_learn_voting.py__
This script creates three bert-based models (RoBERTa, CTBERTv2, and BERT_Large_Cased) and fine-tunes them simultaniously. For each model, the CLS representation is taken and fed to a dense layer to get two score values for the two classes, then these 6 values (2 from each model) are fed to another dense layer that maps them to two final scores for the classes and plays the role of a majority voting mechanism.
The results from this script are not reported in our paper.


