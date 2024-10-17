import argparse

def train_and_test_AE(args):
    from models import AutoEncoder
    print(f"============= Autoencoder / {which_samples} Samples / theta = {args.theta} ===========")
    AE = AutoEncoder(n_feat, 32, args.theta, device, True)
    AE.train(tr_normal, args.n_epochs, tr_anomal, sample_interval=args.sample_interval)
    results = AE.test(test, metrics, anomalies_fraction, {6})
    for metric, result in results.items():
        print(f"{metric}: {result}")
    print(f"Calculated anomaly threshold is {AE.anomaly_threshold}")
    AE.visualize_anomaly_scores(40)
    AE.plot_confusion_matrix()
    while ( answer := input("Do you want to save? [y/n] ") ) not in ["y","n"]: pass
    if answer == "y":
        print(f'Save timestamp = {AE.save(f"AE_saves_{args.dataset}")}')

def train_and_test_VAE(args):
    from models import VariationalAutoEncoder
    print(f"============= Variational Autoencoder / {which_samples} Samples / theta = {args.theta} ===========")
    VAE = VariationalAutoEncoder(n_feat, 32, args.theta, device, True, beta=args.beta, loss_type=args.loss_type,
                                gamma=args.gamma, max_capacity=args.max_capacity, Capacity_max_iter=args.Capacity_max_iter)
    VAE.train(tr_normal, args.n_epochs, tr_anomal, sample_interval=args.sample_interval)
    results = VAE.test(test, metrics, anomalies_fraction, {6})
    for metric, result in results.items():
        print(f"{metric}: {result}")
    print(f"Calculated anomaly threshold is {VAE.anomaly_threshold=}")
    VAE.visualize_anomaly_scores(40)
    VAE.plot_confusion_matrix()
    while ( answer := input("Do you want to save? [y/n] ") ) not in ["y","n"]: pass
    if answer == "y":
        print(f'Save timestamp = {VAE.save(f"VAE_saves_{args.dataset}")}')

def train_and_test_GANomaly(args):
    from models import GANomaly_variant_counterex
    print(f"========== GANomaly_variant / {which_samples} Samples / weights {args.w_adv=}, {args.w_con=}, {args.w_enc=} / theta = {args.theta} ============")
    GAN = GANomaly_variant_counterex(n_feat, 32, 0.2, device, w_adv=args.w_adv, w_con=args.w_con, w_enc=args.w_enc, theta=args.theta)
    GAN.train(tr_normal, args.n_epochs, tr_anomal, n_critic=1, sample_interval=args.sample_interval)
    results = GAN.test(test, metrics, anomalies_fraction, {6})
    for metric, result in results.items():
        print(f"{metric}: {result}")
    print(f"Kai ola auta, me {GAN.anomaly_threshold=}")
    GAN.visualize_anomaly_scores(40)
    GAN.plot_confusion_matrix()
    while ( answer := input("Do you want to save? [y/n] ") ) not in ["y","n"]: pass
    if answer == "y":
        print(f'Save timestamp = {GAN.save(f"GANomaly_saves_{args.dataset}")}')

def train_and_test_BiWGAN_GP(args):
    from models import BiWGAN_GP_counterex
    print(f"============= BiWGAN_GP / {which_samples} Samples / {args.sigma=}, {args.n_critic=} / theta = {args.theta} ==============")
    GAN = BiWGAN_GP_counterex(n_feat, 32, 0.2, device, 10, args.sigma, theta=args.theta)
    GAN.train(tr_normal, args.n_epochs, tr_anomal, n_critic=args.n_critic, sample_interval=args.sample_interval)
    results = GAN.test(test, metrics, anomalies_fraction, {6})
    for metric, result in results.items():
        print(f"{metric}: {result}")
    print(f"Kai ola auta, me {GAN.anomaly_threshold=}")
    GAN.visualize_anomaly_scores(40)
    GAN.plot_confusion_matrix()
    while ( answer := input("Do you want to save? [y/n] ") ) not in ["y","n"]: pass
    if answer == "y":
        print(f'Save timestamp = {GAN.save(f"BiWGAN_GP_saves_{args.dataset}")}')

def train_and_test_ConvAE(args):
    from models import ConvAutoencoder
    print(f"============= ConvAutoencoder / {which_samples} Samples / theta = {args.theta} ==============")
    convAE = ConvAutoencoder(n_features=train_data.get_n_features(), corr_window=args.corr_window_length, n_z=32, n_z_channels=32, device=device, theta=args.theta)
    convAE.train_model(dataset=train_data,num_epochs=args.n_epochs,learning_rate=1e-4, batch_size=64, sample_interval=args.sample_interval)
    results = convAE.test_model(test_data, metrics, 256, anomalies_fraction, {6})
    for metric, result in results.items():
        print(f"{metric}: {result}")
    print(f"Kai ola auta, me {convAE.anomaly_threshold=}")
    convAE.visualize_anomaly_scores(40)
    convAE.plot_confusion_matrix()
    while ( answer := input("Do you want to save? [y/n] ") ) not in ["y","n"]: pass
    if answer == "y":
        print(f'Save timestamp = {convAE.save(f"ConvAE_saves_{args.dataset}")}')
        print("Testing ConvAE")

parser = argparse.ArgumentParser()

# Global parameters
parser.add_argument("--dataset", type=str, choices={"UNSW-NB15","CIC-IDS-2018"}, help="dataset to be used", required=True)
parser.add_argument("--dataset_path", type=str, help="Path to the folder containing the dataset folder", required=True)
parser.add_argument("--n_epochs", type=int, help="number of epochs of training", required=True)
parser.add_argument("--theta", type=float, help="theta value for the hybrid samples", required=True)
parser.add_argument("--sample_interval", type=int, help="every how many batches to display results", required=True)

# Add subparsers for each model
subparsers = parser.add_subparsers(title="models", description="Choose a model", dest="model")

# AutoEncoder-specific parameters
parser_AE = subparsers.add_parser('AE', help='Train and test AutoEncoder')
parser_AE.set_defaults(func=train_and_test_AE)

# Variational AutoEncoder-specific parameters
parser_VAE = subparsers.add_parser('VAE', help='Train and test Variational AutoEncoder')
parser_VAE.add_argument("--loss_type", type=str, choices={"H","B"}, help="type of loss to be used in VAE", required=True)
parser_VAE.add_argument("--beta", type=float, help="beta value for the advanced VAE loss", default=0.5)
parser_VAE.add_argument("--gamma", type=float, help="gamma value for the advanced VAE loss", default=10)
parser_VAE.add_argument("--max_capacity", type=float, help="maximum capacity value for the advanced VAE loss", default=10)
parser_VAE.add_argument("--Capacity_max_iter", type=float, help="maximum iterations for the advanced VAE loss", default=1e5)
parser_VAE.set_defaults(func=train_and_test_VAE)

# GANomaly_variant-specific parameters
parser_GANomaly = subparsers.add_parser('GANomaly_variant', help='Train and test GANomaly_variant')
parser_GANomaly.add_argument("--w_adv", type=float, help="weight of the adversarial loss in GANomaly", required=True)
parser_GANomaly.add_argument("--w_con", type=float, help="weight of the consistency loss in GANomaly", required=True)
parser_GANomaly.add_argument("--w_enc", type=float, help="weight of the encoder loss in GANomaly", required=True)
parser_GANomaly.set_defaults(func=train_and_test_GANomaly)

# ΒiWGAN_GP-specific parameters
parser_BiWGAN_GP = subparsers.add_parser('BiWGAN_GP', help='Train and test BiWGAN_GP')
parser_BiWGAN_GP.add_argument("--n_critic", type=int, help="number of critic iterations in WGAN_GP", required=True)
parser_BiWGAN_GP.add_argument("--sigma", type=float, help="sigma value for the BiWGAN_GP", required=True)
parser_BiWGAN_GP.set_defaults(func=train_and_test_BiWGAN_GP)

# ConvAE-specific parameters
parser_ConvAE = subparsers.add_parser('ConvAE', help='Train and test Convolutional AutoEncoder')
parser_ConvAE.add_argument("--corr_window_length", type=int, help="Length of the correlation window", required=True)
parser_ConvAE.set_defaults(func=train_and_test_ConvAE)

# Parse the arguments
args = parser.parse_args()

import torch
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score, Specificity, AUROC
from datasets import NIDS_Dataset, NORMAL_SAMPLES, ANOMALOUS_SAMPLES, ALL_SAMPLES, PCA

print(f"Loading dataset {args.dataset}...")
if args.model != "ConvAE":
    if args.dataset == "UNSW-NB15":
        train_data_norm = NIDS_Dataset("UNSW-NB15", "UNSW_NB15_training-set.csv", args.dataset_path, which_samples=NORMAL_SAMPLES)
        OH_col_names, normalizer, dict_of_rare_cat_values = train_data_norm.OH_col_names, train_data_norm.normalizer, train_data_norm.dict_of_rare_cat_values
        train_data_anom = NIDS_Dataset("UNSW-NB15", "UNSW_NB15_training-set.csv", args.dataset_path, normalizer=normalizer, do_fit=False, OH_col_names=OH_col_names,
                                    dict_of_rare_cat_values=dict_of_rare_cat_values, which_samples=ANOMALOUS_SAMPLES)
        test_data       = NIDS_Dataset("UNSW-NB15", "UNSW_NB15_testing-set.csv",  args.dataset_path, normalizer=normalizer, do_fit=False, OH_col_names=OH_col_names,
                                    dict_of_rare_cat_values=dict_of_rare_cat_values, which_samples=ALL_SAMPLES)
    elif args.dataset == "CIC-IDS-2018":
        train_data_norm = NIDS_Dataset("CIC-IDS-2018", "CIC-IDS-2018-small-training-set.csv", args.dataset_path, which_samples=NORMAL_SAMPLES)
        OH_col_names, normalizer, dict_of_rare_cat_values = train_data_norm.OH_col_names, train_data_norm.normalizer, train_data_norm.dict_of_rare_cat_values
        train_data_anom = NIDS_Dataset("CIC-IDS-2018", "CIC-IDS-2018-small-training-set.csv", args.dataset_path, normalizer=normalizer,\
                                    do_fit=False, OH_col_names=OH_col_names, dict_of_rare_cat_values=dict_of_rare_cat_values, which_samples=ANOMALOUS_SAMPLES)
        test_data       = NIDS_Dataset("CIC-IDS-2018", "CIC-IDS-2018-small-testing-set.csv", args.dataset_path, normalizer=normalizer,\
                                    do_fit=False, OH_col_names=OH_col_names, dict_of_rare_cat_values=dict_of_rare_cat_values, which_samples=ALL_SAMPLES)
    else:
        pass # We will never get here, argparse will take care of it

    anomalies_fraction = test_data.labels.sum() / len(test_data)
    n_feat = train_data_norm.get_n_features()
    print(f"[+] Dataset loaded successfully! There are {n_feat} features.")
    # Configure data loaders
    tr_normal = torch.utils.data.DataLoader(train_data_norm, batch_size=64, shuffle=True)
    tr_anomal = torch.utils.data.DataLoader(train_data_anom, batch_size=16, shuffle=True)
    test      = torch.utils.data.DataLoader(test_data, batch_size=256)

else:
    if   args.dataset == "UNSW-NB15":
        all_data = NIDS_Dataset("UNSW-NB15", "UNSW_NB15_training-set.csv", args.dataset_path, feature_extractor=PCA, fe_params={"k":32}, which_samples=ALL_SAMPLES, sort_by_pd_col="id")
    elif args.dataset == "CIC-IDS-2018":
        all_data = NIDS_Dataset("CIC-IDS-2018", "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv", args.dataset_path, feature_extractor=PCA, fe_params={"k":32}, which_samples=ALL_SAMPLES, sort_by_pd_col="Timestamp")

    train_data = object.__new__(NIDS_Dataset)
    train_data.features, train_data.labels = all_data.features.clone()[:100_000], all_data.labels.clone()[:100_000]
    test_data  = object.__new__(NIDS_Dataset)
    test_data.features, test_data.labels   = all_data.features.clone()[100_000:200_000], all_data.labels.clone()[100_000:200_000]
    print(f"Training data contain {(anonum := train_data.labels.sum())} anomalous samples and {len(train_data) - anonum} normal ones.")
    print(f"Testing data contain {(anonum := test_data.labels.sum())} anomalous samples and {len(test_data) - anonum} normal ones.")
    anomalies_fraction = anonum / len(test_data)

metrics = [
            ConfusionMatrix(task="binary"),
            Accuracy(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
            F1Score(task="binary"),
            Specificity(task="binary"),
            AUROC(task="binary")
        ]
which_samples = "Normal" if args.theta==0 else "Hybrid"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

args.func(args)