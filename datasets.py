from os.path import join
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.linalg import svd
from torch import tensor

# Placeholder for when normalization is not wanted
class identity_normalizer():
    def __init__(self, norm_columns):
        pass
    def fit(self, data):
        pass
    def transform(self, data):
        pass
    def fit_transform(self, data):
        pass

class z_score_normalizer():
    def __init__(self, norm_columns):
        self.norm_columns = norm_columns
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data[self.norm_columns], axis=0)
        # Calculate std using Bessel's correction
        self.std = np.std(data[self.norm_columns], axis=0, ddof=1)

    def transform(self, data):
        self.std = np.where(self.std == 0, 1, self.std)
        data[self.norm_columns] = (data[self.norm_columns] - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)

class min_max_normalizer():
    def __init__(self, norm_columns):
        self.norm_columns = norm_columns
        self.min = None
        self.max = None

    def fit(self, data):
        data_without_inf = np.where(np.isinf(d := data[self.norm_columns]), np.nan, d)
        self.min = np.nanmin(data_without_inf, axis=0)
        self.max = np.nanmax(data_without_inf, axis=0)

    def transform(self, data):
        max_min = self.max - self.min
        max_min = np.where(max_min == 0, 1, max_min)
        data[self.norm_columns] = (data[self.norm_columns] - self.min) / max_min
        data.replace( np.inf,  5, inplace=True)
        data.replace(-np.inf, -5, inplace=True)

    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)

# PCA feature extractor
class PCA():
    def __init__(self, k):
        self.k = k
    def fit(self, data):
        self.U, self.S, self.V = svd(data,full_matrices=False)
    def get_transformed(self, data):
        return data @ self.V[:,:self.k]

class DatasetParams:
    def __init__(self, label, normal_val, columns, thresholds, drop_columns, read_csv_kwargs):
        self.label = label
        self.normal_val = normal_val
        self.columns = columns
        self.thresholds = thresholds
        self.drop_columns = drop_columns
        self.read_csv_kwargs = read_csv_kwargs

dataset_params = {
    "NSL-KDD": DatasetParams(
        "label", "normal",
        [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label','difficulty_level'
        ],
        {},
        ['difficulty_level'],
        {}
    ),
    "UNSW-NB15": DatasetParams(
        "label", 0, None, {"proto": 250, "state": 2}, ['id','attack_cat'], {}
    ),
    "CIC-IDS-2018": DatasetParams(
        "Label", "Benign", None, {}, ["Timestamp"], {}
    ),
    "BoT-IoT": DatasetParams(
        "attack", 0, None, {}, ["pkSeqID", "saddr", "daddr", "category", "subcategory", "sport", "dport"], {}
    )
}

ALL_SAMPLES = 0
NORMAL_SAMPLES = 1
ANOMALOUS_SAMPLES = 2
class NIDS_Dataset(Dataset):
    def __init__(self, dataset_name, dataset_path, datasets_folder_path="..\datasets", normalizer=min_max_normalizer, feature_extractor=None, fe_params=None, do_fit=True, OH_col_names=None, dict_of_rare_cat_values={}, sort_by_pd_col=None, which_samples=ALL_SAMPLES):
        # Initialize parameters
        self.dataset_name = dataset_name
        self.params = dataset_params[dataset_name]
        self.dataset_path = dataset_path
        self.datasets_folder_path = datasets_folder_path
        self.normalizer = normalizer
        self.feature_extractor = feature_extractor
        self.fe_params = fe_params
        self.normal_only = which_samples
        self.OH_col_names = OH_col_names
        self.dict_of_rare_cat_values = dict_of_rare_cat_values
        self.sort_by_pd_col = sort_by_pd_col

        # Load dataset
        print(f"[*] Initializing dataset {dataset_name}")
        print("[*] Loading Dataset")
        if self.params.columns is None:
            data = pd.read_csv(join(datasets_folder_path, dataset_name, dataset_path), **self.params.read_csv_kwargs)
        else:
            data = pd.read_csv(join(datasets_folder_path, dataset_name, dataset_path),names=self.params.columns, **self.params.read_csv_kwargs)
        # Sanitization
        print("[*] Dropping NaN values")
        data = data.dropna(how='any')
        # Sort if necessary
        if self.sort_by_pd_col is not None:
            print(f"[*] Sorting by {self.sort_by_pd_col}")
            data = data.sort_values(by=self.sort_by_pd_col)
        print(f"[*] Dropping irrelevant columns: {self.params.drop_columns}")
        data = data.drop(self.params.drop_columns, axis=1)
        # Split into features and labels
        print(f"[*] Splitting into features and labels")
        features, labels = data.drop(self.params.label, axis=1), data[self.params.label]
        # Get numeric and categorical columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        categor_cols = features.select_dtypes(exclude=[np.number]).columns
        print("[*] Handling rare categorical values")
        if not self.dict_of_rare_cat_values:
            for col, threshold in self.params.thresholds.items():
                occurrence_dict = features[col].value_counts().to_dict()
                rare_set = {k for k, v in occurrence_dict.items() if v < threshold}
                self.dict_of_rare_cat_values[col] = rare_set
                features[col] = features[col].apply(lambda x: "other" if x in rare_set else x)
        else:
            for col, rare_set in self.dict_of_rare_cat_values.items():
                features[col] = features[col].apply(lambda x: "other" if x in rare_set else x)
        # One hot encode categorical columns
        print("[*] One hot encoding categorical columns")
        features = pd.get_dummies(features, columns=categor_cols)
        # If normal only data are needed, discard all other data
        print("[*] Choosing sample subset (Normal/Anomalous/All)")
        if which_samples == NORMAL_SAMPLES:
            features = features[labels==self.params.normal_val]
        elif which_samples == ANOMALOUS_SAMPLES:
            features = features[labels!=self.params.normal_val]
        # Normalize data
        print("[*] Normalizing data")
        if do_fit:
            self.normalizer = self.normalizer(numeric_cols)
            self.normalizer.fit(features)
        self.normalizer.transform(features)
        # Reindex columns if necessary (ensures consistency between train and test data)
        if self.OH_col_names is not None:
            print("[*] Reindexing columns to avoid inconsistencies between train and test sets")
            features = features.reindex(columns=self.OH_col_names, fill_value=0)
        else:
            self.OH_col_names = features.columns
        # Convert to torch tensors
        print("[*] Converting to torch tensors")
        self.features = tensor(features.to_numpy(dtype=np.float32))
        if which_samples == ALL_SAMPLES:
            self.labels = tensor(labels.apply(lambda x: 0 if x == self.params.normal_val else 1).to_numpy(dtype=np.float32))
        # Apply any feature extraction technique
        if self.feature_extractor is not None:
            print("[*] Performing feature extraction")
            if do_fit:
                self.feature_extractor = self.feature_extractor(**self.fe_params)
                self.feature_extractor.fit(self.features)
            self.features = self.feature_extractor.get_transformed(self.features)
        print("[*] Initialization finished!")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.normal_only:
            return self.features[idx]
        return self.features[idx], self.labels[idx]

    def get_n_features(self):
        return self.features.shape[1]
    
    def keep_subset(self, subset):
        self.features = self.features[subset]
        if not self.normal_only: self.labels = self.labels[subset]