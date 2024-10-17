# Diploma-thesis-NIDS
This repository contains the PyTorch code for my diploma thesis titled: "Training Semi-Supervised Deep Learning Models with Counterexamples for Network Intrusion Detection"

# Purpose of the code
This code has been developed to test a variety of different deep learning models on NIDS datasets, in binary classification mode, i.e. benign / attack. In order to support multiple architectures and datasets, a mini-framework has been developed with rapid development and testing, as well as easy extensibility in mind. Everything is as modular as possible, to maximize code reusability and swapping components for better ones.
This code can be used to test the results provided in my thesis, and extended to include other models and datasets.

# Code structure
## `datasets.py`
Everything related to the datasets and their preprocessing is contained in this file. The core of this file is the `NIDS_Dataset` object. It is a PyTorch `Dataset` child class that can handle all the preprocessing steps necessary for an NIDS dataset. Its capabilities include:
- Taking the name and location of a dataset CSV as input and returning an object that stores the tensors corresponding to the preprocessed features and labels (if there are any).
- Using a custom normalizer (min-max, z-score and identity (none) are implemented).
- Using a custom feature extractor (PCA is implemented as an example).
- Sorting by any feature if temporal information is needed.
- Changing rare categorical feature values ("rare" is defined by user) to "other" to avoid adding One-Hot encoded features unnecessarily.
- Specifying which samples to select (normal, anomalous, or both).
- Performing all of the above for the test set consistently with the training set (normalizer, one-hot column names and dictionary of rare values from the training dataset can be given as input for the test set).
Currently, the datasets CIC-IDS-2018 and UNSW-NB15 have been tested and are fully supported, but partial support for the NSL-KDD and BoT-IoT datasets is provided in this file only (such options are not included in the `main.py` CLI.).

## `all_models.py`
This is where all the model code is stored. For easy and consistent development of models, `ModelBase` was developed. This is a structured parent method from which models can inherit. This takes the mind of the programmer away from standard, repetitive details (the outer training loop of the program, displaying intermediate results to the terminal, enabling and disabling inference modes, properly sending components to devices, saving and loading parameters, plotting results), and allowing them to focus on the important part of the model design. This paradigm also enforces good practices by ensuring that abstract methods are implemented.

In our code, five models are implemented. The use of counterexamples can be controlled through the weight hyperparameter `theta`:
- Classic AutoEncoder 
- Variational AutoEncoder (Two variants: simple $\beta$-VAE (`loss_type` is H) and modified $\beta$-VAE as proposed in [1] (`loss_type` is B))
- GANomaly_variant (a variant of the GANomaly model proposed in [2] and implemented in [this repository](https://github.com/samet-akcay/ganomaly))
- BiWGAN_GP (Our modified implementation of the model proposed in [3] for which no implementation was available)
- A Convolutional AutoEncoder proposed by us. Although this model does not achieve good results, it is included for completeness. *Note that this model does not inherit the `ModelBase` class. This is due to the fact that the architecture differs significantly from the rest of the models, and does not negate the merit of using our `ModelBase` class.*

## `main.py`
This is a CLI developed for the easy training and testing of the included models. Example usage:

### Simple AutoEncoder
```bash
python .\get_model_results.py --dataset CIC-IDS-2018 --n_epochs 20 --theta 0.001 --sample_interval 1000 AE
```
### Variational AutoEncoder
#### Simple
```bash
python .\get_model_results.py --dataset CIC-IDS-2018 --n_epochs 20 --theta 0.001 --sample_interval 1000 VAE --loss_type H --beta 0.5
```
#### Modified
```bash
python .\get_model_results.py --dataset CIC-IDS-2018 --n_epochs 20 --theta 0.001 --sample_interval 1000 VAE --loss_type B --gamma 10 --max_capacity 10 --Capacity_max_iter 1e5
```
### GANomaly_variant
```bash
python .\get_model_results.py --dataset CIC-IDS-2018 --n_epochs 20 --theta 0.001 --sample_interval 1000 GANomaly_variant --w_adv 1 --w_con 50 --w_enc 1
```
### BiWGAN_GP
```bash
python .\get_model_results.py --dataset CIC-IDS-2018 --n_epochs 20 --theta 0.001 --sample_interval 1000 BiWGAN_GP --n_critic 5 --sigma 10
```
### Convolutional AutoEncoder
```bash
python .\main.py --dataset CIC-IDS-2018 --n_epochs 20 --theta 0.001 --sample_interval 1000 ConvAE --corr_window_length 5
```

# References
[1] C.P.Burgess, I.Higgins, A.Pal, L.Matthey, N.Watters, G.Desjardins and A.Lerchner, "Understanding disentangling in $\beta$-VAE", 2018.

[2] S.Akcay, A.Atapour-Abarghouei and T.P.Breckon, "GANomaly: Semi-supervised Anomaly Detection via Adversarial Training" in Computer Vision – ACCV 2018, Cham, 2019.

[3] W.Yao, H.Shi and H.Zhao, "Scalable anomaly-based intrusion detection for secure Internet of Things using generative adversarial networks in fog environment" Journal of Network and Computer Applications, vol. 214, p. 103622, 2023.

# Copyright
Copyright © Markos P. Deligiannis, 2024. All rights reserved.
