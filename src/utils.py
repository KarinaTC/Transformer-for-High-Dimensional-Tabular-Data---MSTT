from torch.utils.data import Dataset

class CustomDictDataset(Dataset):
    """
    Custom PyTorch Dataset for tabular classification.

    Converts a dictionary containing preprocessed numerical
    and/or categorical features and target labels into a
    PyTorch-compatible Dataset.Numerical features are converted to float,
    while categorical features are converted to long to support embedding layers.

    The expected dictionary structure is:

        {
            "y": target_labels,
            "x_cont": numerical_features,      # Optional
            "x_cat": categorical_features      # Optional
        }

    Parameters:
    - data_dict (dict): Dictionary containing the preprocessed features and 
      target labels as PyTorch tensors.
    - binary_class (bool): Default = True. If True, converts target labels 
      to float for binary classification. Otherwise, converts them to long
      for multiclass classification.

    Return:
    - The Dataset returns each sample as a dictionary containing the target label and the available feature types.
    """
    def __init__(self, data_dict, binary_class=True):

        # Labels (ya son tensores)
        if binary_class:
            self.labels = data_dict["y"].float()
        else:
            self.labels = data_dict["y"].long()

        # Categorical features
        self.has_categorical = "x_cat" in data_dict
        if self.has_categorical:
            self.categorical = data_dict["x_cat"].long()

        # Continuous features
        self.has_numerical = "x_cont" in data_dict
        if self.has_numerical:
            self.continuous = data_dict["x_cont"].float()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        sample = {"target": self.labels[idx]}
        if self.has_categorical:
            sample["x_cat"] = self.categorical[idx]
        if self.has_numerical:
            sample["x_cont"] = self.continuous[idx]
        return sample